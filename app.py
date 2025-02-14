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

class FaceDetector:
    def __init__(self):
        # إعداد MediaPipe Face Detection مع معايير صارمة
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7  # زيادة دقة الكشف
        )
        
        # إعداد MediaPipe Face Mesh للتأكد من الوجوه
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.7
        )

    def is_valid_face(self, box: list, image_shape: tuple) -> bool:
        """التحقق من صحة الوجه المكتشف"""
        x, y, w, h = box
        height, width = image_shape[:2]
        
        # التحقق من نسبة أبعاد الوجه
        aspect_ratio = w / h
        if not (0.6 <= aspect_ratio <= 1.4):  # نسبة الوجه البشري الطبيعية
            return False
        
        # التحقق من حجم الوجه بالنسبة للصورة
        face_area = w * h
        image_area = width * height
        face_area_ratio = face_area / image_area
        
        if face_area_ratio < 0.01 or face_area_ratio > 0.4:
            return False
            
        return True

    def detect_faces(self, image: np.ndarray) -> list:
        faces = []
        height, width = image.shape[:2]
        
        # تحويل الصورة إلى RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # كشف الوجوه باستخدام MediaPipe
        results = self.mp_face.process(rgb_image)
        
        if results.detections:
            for detection in results.detections:
                if detection.score[0] > 0.7:  # التحقق من مستوى الثقة
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * width))
                    y = max(0, int(bbox.ymin * height))
                    w = min(int(bbox.width * width), width - x)
                    h = min(int(bbox.height * height), height - y)
                    
                    # التحقق من صحة الوجه
                    if self.is_valid_face([x, y, w, h], image.shape):
                        # توسيع منطقة الوجه قليلاً
                        padding_x = int(w * 0.1)
                        padding_y = int(h * 0.1)
                        x = max(0, x - padding_x)
                        y = max(0, y - padding_y)
                        w = min(w + 2*padding_x, width - x)
                        h = min(h + 2*padding_y, height - y)
                        
                        faces.append({
                            'box': [x, y, w, h],
                            'confidence': float(detection.score[0])
                        })
        
        return faces

class FaceBlurProcessor:
    def __init__(self):
        self.detector = FaceDetector()
    
    def apply_blur(self, image: np.ndarray, box: list) -> np.ndarray:
        x, y, w, h = box
        center = (x + w//2, y + h//2)
        radius = int(max(w, h) * 0.9)  # زيادة منطقة التمويه
        
        # إنشاء قناع دائري متدرج
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.circle(mask, center, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (99, 99), 30)
        
        # تطبيق تمويه قوي جداً
        blurred = cv2.GaussianBlur(image, (99, 99), 30)
        blurred = cv2.GaussianBlur(blurred, (99, 99), 30)  # تمويه مضاعف
        
        # دمج الصور
        mask = np.expand_dims(mask, -1)
        result = image * (1 - mask) + blurred * mask
        
        return result.astype(np.uint8)
    
    def process_image(self, image: Image.Image) -> Image.Image:
        try:
            img = np.array(image)
            faces = self.detector.detect_faces(img)
            
            if not faces:
                return image
            
            for face in faces:
                img = self.apply_blur(img, face['box'])
            
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
        "اختر صورة أو ملف PDF",
        type=["jpg", "jpeg", "png", "pdf"]
    )
    
    if uploaded_file:
        try:
            with st.spinner("جاري المعالجة..."):
                if uploaded_file.type == "application/pdf":
                    images = convert_from_bytes(uploaded_file.read())
                    for idx, img in enumerate(images, 1):
                        processed = processor.process_image(img)
                        st.image(processed, caption=f"الصفحة {idx}", use_column_width=True)
                        
                        buf = io.BytesIO()
                        processed.save(buf, format="PNG")
                        st.download_button(
                            f"تحميل الصفحة {idx}",
                            buf.getvalue(),
                            f"processed_page_{idx}.png",
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
                        "تحميل الصورة المعالجة",
                        buf.getvalue(),
                        "processed_image.png",
                        "image/png"
                    )
            
            st.success("تمت المعالجة بنجاح!")
            
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    main()
