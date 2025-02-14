import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from pdf2image import convert_from_bytes
from PIL import Image
import io
import logging
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_custom_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    :root {
        --primary-color: #1E3D59;
        --secondary-color: #FF6E40;
        --accent-color: #17B794;
        --background-color: #F5F5F5;
        --card-bg: rgba(255, 255, 255, 0.95);
    }
    
    * {
        font-family: 'Tajawal', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1E3D59 0%, #17B794 100%);
    }
    
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .custom-card {
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        animation: fadeIn 0.5s ease;
    }
    
    .title-card {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 3rem;
    }
    
    .title-card h1 {
        color: var(--primary-color);
        font-size: 3rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .title-card p {
        color: var(--secondary-color);
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #17B794 0%, #1E3D59 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .upload-card {
        background: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 2rem 0;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        border: none;
        font-weight: 500;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(23, 183, 148, 0.3);
    }
    
    .success-message {
        padding: 1rem;
        background: #17B794;
        color: white;
        border-radius: 10px;
        text-align: center;
        animation: slideIn 0.5s ease;
    }
    
    .error-message {
        padding: 1rem;
        background: #FF6E40;
        color: white;
        border-radius: 10px;
        text-align: center;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    .progress-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stProgress > div > div {
        background-color: var(--accent-color);
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
        self.processed_count = 0
    
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
    
    def process_image(self, image: Image.Image) -> tuple:
        try:
            img = np.array(image)
            faces = self.detector.detect_faces(img)
            
            if faces:
                for face in faces:
                    img = self.apply_blur(img, face['box'])
                self.processed_count += len(faces)
            
            return Image.fromarray(img), len(faces)
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {str(e)}")
            raise

def main():
    set_custom_style()
    
    st.markdown("""
    <div class="title-card">
        <h1>🎭 نظام تمويه الوجوه الذكي</h1>
        <p>معالجة متطورة للصور باستخدام الذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    processor = FaceBlurProcessor()
    
    st.markdown("""
    <div class="upload-card">
        <h2>📤 رفع الملفات</h2>
        <p>يمكنك رفع صور بصيغة JPG, JPEG, PNG أو ملفات PDF</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "pdf"]
    )
    
    if uploaded_file:
        try:
            with st.spinner("جاري المعالجة..."):
                progress = st.progress(0)
                status = st.empty()
                
                if uploaded_file.type == "application/pdf":
                    images = convert_from_bytes(uploaded_file.read())
                    total_faces = 0
                    
                    for idx, img in enumerate(images, 1):
                        progress.progress((idx / len(images)))
                        status.text(f"معالجة الصفحة {idx} من {len(images)}")
                        
                        processed, faces_count = processor.process_image(img)
                        total_faces += faces_count
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>نتيجة معالجة الصفحة {idx}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.image(processed, use_column_width=True)
                        
                        buf = io.BytesIO()
                        processed.save(buf, format="PNG")
                        st.download_button(
                            f"تحميل الصفحة {idx}",
                            buf.getvalue(),
                            f"processed_page_{idx}.png",
                            "image/png"
                        )
                    
                    st.markdown(f"""
                    <div class="stats-card">
                        <h3>إحصائيات المعالجة</h3>
                        <p>عدد الصفحات: {len(images)}</p>
                        <p>إجمالي الوجوه المكتشفة: {total_faces}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    image = Image.open(uploaded_file)
                    processed, faces_count = processor.process_image(image)
                    
                    st.markdown("""
                    <div class="result-card">
                        <h3>نتيجة المعالجة</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="الصورة الأصلية", use_column_width=True)
                    with col2:
                        st.image(processed, caption="الصورة بعد المعالجة", use_column_width=True)
                    
                    st.markdown(f"""
                    <div class="stats-card">
                        <h3>إحصائيات المعالجة</h3>
                        <p>الوجوه المكتشفة: {faces_count}</p>
                        <p>تاريخ المعالجة: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    buf = io.BytesIO()
                    processed.save(buf, format="PNG")
                    st.download_button(
                        "تحميل الصورة المعالجة",
                        buf.getvalue(),
                        "processed_image.png",
                        "image/png"
                    )
                
                st.markdown("""
                <div class="success-message">
                    <h3>✨ تمت المعالجة بنجاح!</h3>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            
        except Exception as e:
            st.markdown(f"""
            <div class="error-message">
                <h3>❌ حدث خطأ</h3>
                <p>{str(e)}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
