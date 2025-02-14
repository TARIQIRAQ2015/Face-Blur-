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
        # تحسين حساسية الكشف
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.2  # زيادة الحساسية للوجوه الصغيرة
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=50,  # زيادة عدد الوجوه المكتشفة
            refine_landmarks=True,
            min_detection_confidence=0.2
        )
        
        # تحسين كشف الوجوه الصغيرة
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    def detect_faces(self, image: np.ndarray) -> list:
        faces = []
        height, width = image.shape[:2]
        
        # تحسين جودة الصورة للكشف
        enhanced = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)  # زيادة التباين
        
        # كشف الوجوه بأحجام مختلفة
        scales = [1.0, 1.5, 2.0]  # تكبير الصورة للكشف عن الوجوه الصغيرة
        for scale in scales:
            if scale != 1.0:
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                scaled_image = cv2.resize(enhanced, (scaled_width, scaled_height))
            else:
                scaled_image = enhanced
            
            # MediaPipe Detection
            rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(rgb_image)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * scaled_width / scale))
                    y = max(0, int(bbox.ymin * scaled_height / scale))
                    w = min(int(bbox.width * scaled_width / scale), width - x)
                    h = min(int(bbox.height * scaled_height / scale), height - y)
                    faces.append({
                        'box': [x, y, w, h],
                        'confidence': detection.score[0],
                        'source': f'mediapipe_scale_{scale}'
                    })
            
            # Haar Cascade Detection
            gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # تحسين معامل الكشف للوجوه الصغيرة
            front_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # تقليل للكشف عن الوجوه الصغيرة
                minNeighbors=3,    # تقليل للحساسية العالية
                minSize=(15, 15)   # تقليل الحجم الأدنى
            )
            
            for (x, y, w, h) in front_faces:
                faces.append({
                    'box': [
                        int(x/scale), int(y/scale),
                        int(w/scale), int(h/scale)
                    ],
                    'confidence': 0.8,
                    'source': f'haar_scale_{scale}'
                })
        
        return self.merge_detections(faces, width, height)
    
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
            
            for existing_face in final_faces:
                box2 = existing_face['box']
                iou = self.calculate_iou(box1, box2)
                if iou > 0.3:  # خفض عتبة التداخل لتجنب التكرار
                    if face['confidence'] > existing_face['confidence']:
                        final_faces.remove(existing_face)
                    else:
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
    
    def apply_strong_blur(self, image: np.ndarray, box: list, confidence: float) -> np.ndarray:
        """تطبيق تمويه قوي جداً"""
        x, y, w, h = box
        center = (x + w//2, y + h//2)
        radius = int(max(w, h) * 0.7)  # زيادة منطقة التمويه
        
        # إنشاء قناع متدرج قوي
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        for r in range(radius):
            alpha = 1 - (r / radius)**1.5  # تدرج أقوى
            cv2.circle(mask, center, radius - r, alpha, 1)
        
        # تطبيق عدة مستويات من التمويه القوي
        blur_levels = [
            cv2.GaussianBlur(image, (k, k), 0)
            for k in [41, 81, 121, 161]  # زيادة قوة التمويه
        ]
        
        # إضافة تمويه إضافي للمنطقة المركزية
        extra_blur = cv2.GaussianBlur(image, (201, 201), 0)
        center_mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.circle(center_mask, center, int(radius * 0.7), 1.0, -1)
        center_mask = cv2.GaussianBlur(center_mask, (41, 41), 0)
        
        result = image.copy()
        # تطبيق التمويه المتدرج
        for i, blurred in enumerate(blur_levels):
            weight = mask * (1 - i/len(blur_levels))
            weight = np.expand_dims(weight, -1)
            result = result * (1 - weight) + blurred * weight
        
        # إضافة التمويه المركزي القوي
        center_mask = np.expand_dims(center_mask, -1)
        result = result * (1 - center_mask) + extra_blur * center_mask
        
        return result.astype(np.uint8)
    
    def process_image(self, image: Image.Image) -> Image.Image:
        try:
            img = np.array(image)
            # تحسين جودة الصورة
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # كشف الوجوه
            faces = self.detector.detect_faces(img)
            
            if not faces:
                return image
            
            # تمويه كل وجه
            for face in faces:
                img = self.apply_strong_blur(img, face['box'], face['confidence'])
            
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
