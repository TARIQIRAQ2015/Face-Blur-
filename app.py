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
        # تكوين كاشف MediaPipe للوجوه
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # نموذج للمدى البعيد
            min_detection_confidence=0.15  # حساسية عالية للكشف
        )
        
        # تكوين كاشف Haar للوجوه
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """تحسين جودة الصورة للكشف عن الوجوه"""
        # تحسين التباين
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # تقليل الضوضاء
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        return enhanced

    def detect_faces(self, image: np.ndarray) -> list:
        faces = []
        original_height, original_width = image.shape[:2]
        
        # تحسين جودة الصورة
        enhanced = self.enhance_image(image)
        
        # قائمة المقاييس للكشف عن الوجوه بأحجام مختلفة
        scales = [0.8, 1.0, 1.5, 2.0]  # إضافة مقياس أصغر للوجوه الكبيرة
        
        for scale in scales:
            current_width = int(original_width * scale)
            current_height = int(original_height * scale)
            
            # تغيير حجم الصورة للمقياس الحالي
            if scale != 1.0:
                current_image = cv2.resize(enhanced, (current_width, current_height))
            else:
                current_image = enhanced
            
            # MediaPipe Detection
            rgb_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(rgb_image)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * current_width))
                    y = max(0, int(bbox.ymin * current_height))
                    w = min(int(bbox.width * current_width), current_width - x)
                    h = min(int(bbox.height * current_height), current_height - y)
                    
                    # تحويل الإحداثيات للمقياس الأصلي
                    scaled_x = int(x / scale)
                    scaled_y = int(y / scale)
                    scaled_w = int(w / scale)
                    scaled_h = int(h / scale)
                    
                    # توسيع منطقة الوجه قليلاً
                    padding = 0.1  # 10% padding
                    pad_x = int(scaled_w * padding)
                    pad_y = int(scaled_h * padding)
                    
                    scaled_x = max(0, scaled_x - pad_x)
                    scaled_y = max(0, scaled_y - pad_y)
                    scaled_w = min(scaled_w + 2*pad_x, original_width - scaled_x)
                    scaled_h = min(scaled_h + 2*pad_y, original_height - scaled_y)
                    
                    faces.append({
                        'box': [scaled_x, scaled_y, scaled_w, scaled_h],
                        'confidence': detection.score[0],
                        'source': 'mediapipe'
                    })
            
            # Haar Cascade Detection
            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # كشف الوجوه الأمامية
            front_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
            
            # كشف الوجوه الجانبية
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
            
            # دمج نتائج الكشف
            for (x, y, w, h) in np.vstack((front_faces, profile_faces)) if len(profile_faces) > 0 else front_faces:
                scaled_x = int(x / scale)
                scaled_y = int(y / scale)
                scaled_w = int(w / scale)
                scaled_h = int(h / scale)
                
                # توسيع منطقة الوجه
                padding = 0.1
                pad_x = int(scaled_w * padding)
                pad_y = int(scaled_h * padding)
                
                scaled_x = max(0, scaled_x - pad_x)
                scaled_y = max(0, scaled_y - pad_y)
                scaled_w = min(scaled_w + 2*pad_x, original_width - scaled_x)
                scaled_h = min(scaled_h + 2*pad_y, original_height - scaled_y)
                
                faces.append({
                    'box': [scaled_x, scaled_y, scaled_w, scaled_h],
                    'confidence': 0.8,
                    'source': 'haar'
                })
        
        return self.merge_detections(faces, original_width, original_height)
    
    def merge_detections(self, faces: list, width: int, height: int) -> list:
        if not faces:
            return []
        
        # ترتيب النتائج حسب مستوى الثقة
        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
        final_faces = []
        
        for face in faces:
            should_add = True
            box1 = face['box']
            
            # تجاهل المربعات الصغيرة جداً أو الكبيرة جداً
            area = box1[2] * box1[3]
            total_area = width * height
            if area < (total_area * 0.001) or area > (total_area * 0.5):
                continue
            
            # التحقق من التداخل مع الوجوه الموجودة
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
        radius = int(max(w, h) * 0.7)
        
        # إنشاء قناع متدرج للتمويه
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.circle(mask, center, radius, 1.0, -1)
        
        # تطبيق تمويه متعدد المستويات
        blur_levels = [
            cv2.GaussianBlur(image, (k, k), 0)
            for k in [51, 99, 151, 201]
        ]
        
        # دمج مستويات التمويه
        result = image.copy()
        for i, blurred in enumerate(blur_levels):
            weight = cv2.GaussianBlur(mask, (99, 99), 30) * (1 - i/len(blur_levels))
            weight = np.expand_dims(weight, -1)
            result = result * (1 - weight) + blurred * weight
        
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
