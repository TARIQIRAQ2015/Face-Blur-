import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from pdf2image import convert_from_bytes
from PIL import Image
import io
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_page_config():
    st.set_page_config(
        page_title="تمويه الوجوه الذكي",
        page_icon="🎭",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(120deg, #E3F2FD 0%, #BBDEFB 100%);
    }
    
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #2196F3, #0039CB);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        width: 100%;
        transition: transform 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

class AdvancedFaceDetector:
    def __init__(self):
        # تكوين كاشف MediaPipe للوجوه البشرية فقط
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # نموذج للمدى البعيد
            min_detection_confidence=0.6  # زيادة الثقة لتجنب الكشف الخاطئ
        )
        
        # تكوين كاشف Haar للوجوه البشرية
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def filter_face_detections(self, box: list, image_shape: tuple) -> bool:
        """التحقق من أن المنطقة المكتشفة هي وجه بشري"""
        x, y, w, h = box
        height, width = image_shape[:2]
        
        # نسبة العرض إلى الارتفاع للوجه البشري
        aspect_ratio = w / h
        if not (0.5 <= aspect_ratio <= 1.5):  # الوجوه البشرية عادة قريبة من المربع
            return False
        
        # حجم الوجه بالنسبة للصورة
        face_area = w * h
        image_area = width * height
        face_area_ratio = face_area / image_area
        
        # تجاهل المناطق الصغيرة جداً أو الكبيرة جداً
        if face_area_ratio < 0.01 or face_area_ratio > 0.5:
            return False
        
        return True

    def detect_faces(self, image: np.ndarray) -> list:
        faces = []
        height, width = image.shape[:2]
        
        # MediaPipe Detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb_image)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * width))
                y = max(0, int(bbox.ymin * height))
                w = min(int(bbox.width * width), width - x)
                h = min(int(bbox.height * height), height - y)
                
                # التحقق من أن المنطقة المكتشفة هي وجه بشري
                if self.filter_face_detections([x, y, w, h], image.shape):
                    faces.append({
                        'box': [x, y, w, h],
                        'confidence': detection.score[0],
                        'source': 'mediapipe'
                    })
        
        # Haar Cascade Detection كاحتياطي
        if not faces:  # فقط إذا لم يجد MediaPipe أي وجوه
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            detected_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,  # زيادة للتقليل من الكشف الخاطئ
                minSize=(30, 30)  # زيادة الحجم الأدنى
            )
            
            for (x, y, w, h) in detected_faces:
                if self.filter_face_detections([x, y, w, h], image.shape):
                    faces.append({
                        'box': [x, y, w, h],
                        'confidence': 0.8,
                        'source': 'haar'
                    })
        
        return self.merge_detections(faces)
    
    def merge_detections(self, faces: list) -> list:
        if not faces:
            return []
        
        # ترتيب النتائج حسب مستوى الثقة
        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
        final_faces = []
        
        for face in faces:
            should_add = True
            box1 = face['box']
            
            for existing_face in final_faces:
                box2 = existing_face['box']
                iou = self.calculate_iou(box1, box2)
                if iou > 0.3:
                    should_add = False
                    break
            
            if should_add:
                final_faces.append(face)
        
        return final_faces
    
    @staticmethod
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

class FaceBlurProcessor:
    def __init__(self):
        self.detector = AdvancedFaceDetector()
    
    def apply_strong_blur(self, image: np.ndarray, box: list) -> np.ndarray:
        x, y, w, h = box
        center = (x + w//2, y + h//2)
        radius = int(max(w, h) * 0.8)  # زيادة منطقة التمويه
        
        # إنشاء قناع متدرج للتمويه
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.circle(mask, center, radius, 1.0, -1)
        
        # تطبيق تمويه قوي جداً متعدد المستويات
        blur_levels = [
            cv2.GaussianBlur(image, (k, k), 0)
            for k in [99, 151, 201]  # زيادة قوة التمويه
        ]
        
        result = image.copy()
        for i, blurred in enumerate(blur_levels):
            weight = cv2.GaussianBlur(mask, (151, 151), 50) * (1 - i/len(blur_levels))
            weight = np.expand_dims(weight, -1)
            result = result * (1 - weight) + blurred * weight
        
        # تطبيق تمويه إضافي للتأكد من إخفاء الملامح
        final_blur = cv2.GaussianBlur(result, (201, 201), 60)
        final_weight = cv2.GaussianBlur(mask, (201, 201), 60)
        final_weight = np.expand_dims(final_weight, -1)
        result = result * (1 - final_weight * 0.5) + final_blur * (final_weight * 0.5)
        
        return result.astype(np.uint8)
    
    def process_image(self, image: Image.Image) -> Image.Image:
        try:
            img = np.array(image)
            faces = self.detector.detect_faces(img)
            
            if not faces:
                return image
            
            # تمويه كل وجه
            for face in faces:
                img = self.apply_strong_blur(img, face['box'])
            
            logger.info(f"تم العثور على {len(faces)} وجه/وجوه")
            return Image.fromarray(img)
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {str(e)}")
            raise

def main():
    set_page_config()
    
    st.title("🎭 تمويه الوجوه الذكي")
    st.markdown("""
    <p style='font-size: 1.2rem; text-align: center;'>
        نظام متقدم للكشف عن الوجوه وتمويهها باستخدام الذكاء الاصطناعي
    </p>
    """, unsafe_allow_html=True)
    
    processor = FaceBlurProcessor()
    
    uploaded_file = st.file_uploader(
        "اختر ملفاً",
        type=["jpg", "jpeg", "png", "pdf"]
    )
    
    if uploaded_file:
        try:
            with st.spinner("🔄 جاري المعالجة..."):
                if uploaded_file.type == "application/pdf":
                    images = convert_from_bytes(uploaded_file.read())
                    for idx, img in enumerate(images, 1):
                        processed = processor.process_image(img)
                        
                        st.subheader(f"📄 صفحة {idx}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(processed, use_column_width=True)
                        with col2:
                            buf = io.BytesIO()
                            processed.save(buf, format="PNG")
                            st.download_button(
                                f"⬇️ تحميل الصفحة {idx}",
                                buf.getvalue(),
                                f"blurred_page_{idx}.png",
                                "image/png"
                            )
                else:
                    image = Image.open(uploaded_file)
                    processed = processor.process_image(image)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="الصورة الأصلية", use_column_width=True)
                    with col2:
                        st.image(processed, caption="الصورة بعد المعالجة", use_column_width=True)
                        
                        buf = io.BytesIO()
                        processed.save(buf, format="PNG")
                        st.download_button(
                            "⬇️ تحميل الصورة المعدلة",
                            buf.getvalue(),
                            "blurred_image.png",
                            "image/png"
                        )
            
            st.success("✨ تمت المعالجة بنجاح!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ حدث خطأ: {str(e)}")

if __name__ == "__main__":
    main()
