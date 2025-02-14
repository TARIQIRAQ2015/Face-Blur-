import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import io
import logging
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_luxury_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&display=swap');
    
    :root {
        --primary-color: #14213D;
        --secondary-color: #FCA311;
        --accent-color: #E5E5E5;
        --gold: #FFD700;
        --dark-gold: #DAA520;
    }
    
    * {
        font-family: 'Tajawal', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--primary-color) 0%, #000000 100%);
    }
    
    .luxury-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .luxury-title {
        text-align: center;
        padding: 3rem;
        background: rgba(20, 33, 61, 0.95);
        border: 2px solid var(--gold);
        border-radius: 20px;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
        margin-bottom: 3rem;
        backdrop-filter: blur(10px);
    }
    
    .luxury-title h1 {
        color: var(--gold);
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .luxury-title p {
        color: var(--accent-color);
        font-size: 1.4rem;
        font-weight: 500;
    }
    
    .upload-zone {
        background: rgba(20, 33, 61, 0.95);
        border: 2px solid var(--gold);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
    }
    
    .stats-panel {
        background: linear-gradient(45deg, var(--primary-color), #000000);
        border: 1px solid var(--gold);
        color: var(--accent-color);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .result-panel {
        background: rgba(20, 33, 61, 0.95);
        border: 2px solid var(--gold);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    .download-btn {
        background: linear-gradient(45deg, var(--gold), var(--dark-gold));
        color: var(--primary-color);
        padding: 1rem 2rem;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        width: 100%;
        margin-top: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.3);
    }
    
    .progress-bar {
        height: 10px;
        background: rgba(255, 215, 0, 0.1);
        border-radius: 5px;
        overflow: hidden;
    }
    
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--gold), var(--dark-gold));
        transition: width 0.3s ease;
    }
    
    .success-msg {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: slideUp 0.5s ease;
    }
    
    .error-msg {
        background: linear-gradient(45deg, #dc3545, #c82333);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: slideUp 0.5s ease;
    }
    
    @keyframes slideUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* تحسين شكل الأزرار */
    .stButton>button {
        background: linear-gradient(45deg, var(--gold), var(--dark-gold));
        color: var(--primary-color);
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

class SimpleFaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    def filter_detection(self, x, y, w, h, image_shape):
        """فلترة النتائج لتجنب النصوص والأشكال غير المطلوبة"""
        area = w * h
        image_area = image_shape[0] * image_shape[1]
        aspect_ratio = w / h
        
        # تعديل نسبة المساحة المقبولة
        min_area_ratio = 0.005  # تقليل الحد الأدنى للمساحة للوجوه الصغيرة
        max_area_ratio = 0.3    # زيادة الحد الأقصى للوجوه الكبيرة
        
        if area < (image_area * min_area_ratio) or area > (image_area * max_area_ratio):
            return False
            
        # تحسين نسب الوجه
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:  # نسب أكثر دقة للوجوه
            return False
            
        # تعديل الحد الأدنى للأبعاد
        if w < 20 or h < 20:  # تقليل الحد الأدنى للوجوه الصغيرة
            return False
            
        return True

    def detect_faces(self, image: np.ndarray) -> list:
        faces = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تحسين معايير الكشف
        scales = [1.05, 1.1, 1.15]  # استخدام عدة مقاييس للكشف
        for scale in scales:
            # كشف الوجوه الأمامية
            front_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=4,  # تقليل للكشف عن الوجوه الصغيرة
                minSize=(20, 20),  # تقليل الحد الأدنى
                maxSize=(300, 300),  # تحديد الحد الأقصى
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # إضافة الوجوه المكتشفة
            for (x, y, w, h) in front_faces:
                if self.filter_detection(x, y, w, h, image.shape):
                    # تعديل حجم منطقة التمويه
                    padding = 0.08 if w < 50 else 0.15  # padding أقل للوجوه الصغيرة
                    padding_x = int(w * padding)
                    padding_y = int(h * padding)
                    
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    w = min(w + 2*padding_x, image.shape[1] - x)
                    h = min(h + 2*padding_y, image.shape[0] - y)
                    
                    faces.append({
                        'box': [x, y, w, h],
                        'confidence': 0.9
                    })
        
        return faces

class FaceBlurProcessor:
    def __init__(self):
        self.detector = SimpleFaceDetector()
    
    def create_circular_mask(self, image_shape, center, radius):
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        y, x = np.ogrid[:image_shape[0], :image_shape[1]]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # تحسين التدرج للتمويه
        mask = np.clip(1 - dist_from_center/radius, 0, 1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)  # تنعيم أقل للحواف
        return mask
    
    def apply_circular_blur(self, image: np.ndarray, box: list) -> np.ndarray:
        x, y, w, h = box
        center = (x + w//2, y + h//2)
        
        # تعديل نصف القطر حسب حجم الوجه
        radius = int(max(w, h) * (0.5 if w < 50 else 0.65))
        
        # تعديل قوة التمويه حسب حجم الوجه
        blur_size = 51 if w < 50 else 99
        blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 30)
        
        mask = self.create_circular_mask(image.shape, center, radius)
        mask = np.expand_dims(mask, axis=2)
        
        result = image.copy()
        result = image * (1 - mask) + blurred * mask
        
        return result.astype(np.uint8)
    
    def process_image(self, image: Image.Image) -> tuple:
        try:
            img = np.array(image)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            faces = self.detector.detect_faces(img)
            
            if faces:
                for face in faces:
                    img = self.apply_circular_blur(img, face['box'])
            
            return Image.fromarray(img), len(faces)
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {str(e)}")
            raise

def main():
    set_luxury_style()
    
    st.markdown("""
    <div class="luxury-title">
        <h1>🎭 نظام تمويه الوجوه الاحترافي</h1>
        <p>معالجة فائقة الدقة باستخدام أحدث تقنيات الذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    processor = FaceBlurProcessor()
    
    st.markdown("""
    <div class="upload-zone">
        <h2 style='color: var(--gold); font-size: 2rem; margin-bottom: 1rem;'>📤 رفع الملفات</h2>
        <p style='color: var(--accent-color); font-size: 1.2rem;'>يمكنك رفع صور بصيغة JPG, JPEG, PNG أو ملفات PDF</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file:
        try:
            with st.spinner("جاري المعالجة..."):
                progress = st.progress(0)
                status = st.empty()
                
                if uploaded_file.type == "application/pdf":
                    images = convert_from_bytes(uploaded_file.read())
                    total_faces = 0
                    processed_images = []
                    
                    for idx, img in enumerate(images, 1):
                        progress.progress((idx / len(images)))
                        status.text(f"معالجة الصفحة {idx} من {len(images)}")
                        
                        processed, faces_count = processor.process_image(img)
                        total_faces += faces_count
                        processed_images.append(processed)
                        
                        st.markdown(f"""
                        <div class="result-panel">
                            <h3 style='color: var(--gold);'>نتيجة معالجة الصفحة {idx}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.image(processed, use_column_width=True)
                    
                    # تجميع كل الصور في ملف PDF واحد
                    pdf_buffer = io.BytesIO()
                    processed_images[0].save(
                        pdf_buffer, "PDF", save_all=True, 
                        append_images=processed_images[1:]
                    )
                    
                    st.markdown(f"""
                    <div class="stats-panel">
                        <h3 style='color: var(--gold);'>إحصائيات المعالجة</h3>
                        <p>عدد الصفحات: {len(images)}</p>
                        <p>إجمالي الوجوه المكتشفة: {total_faces}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        "⬇️ تحميل الملف كاملاً",
                        pdf_buffer.getvalue(),
                        "processed_document.pdf",
                        "application/pdf"
                    )
                    
                else:
                    image = Image.open(uploaded_file)
                    processed, faces_count = processor.process_image(image)
                    
                    st.markdown("""
                    <div class="result-panel">
                        <h3 style='color: var(--gold);'>نتيجة المعالجة</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="الصورة الأصلية", use_column_width=True)
                    with col2:
                        st.image(processed, caption="الصورة بعد المعالجة", use_column_width=True)
                    
                    st.markdown(f"""
                    <div class="stats-panel">
                        <h3 style='color: var(--gold);'>إحصائيات المعالجة</h3>
                        <p>الوجوه المكتشفة: {faces_count}</p>
                        <p>تاريخ المعالجة: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    buf = io.BytesIO()
                    processed.save(buf, format="PNG")
                    st.download_button(
                        "⬇️ تحميل الصورة المعالجة",
                        buf.getvalue(),
                        "processed_image.png",
                        "image/png"
                    )
                
                st.markdown("""
                <div class="success-msg">
                    <h3>✨ تمت المعالجة بنجاح!</h3>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            
        except Exception as e:
            st.markdown(f"""
            <div class="error-msg">
                <h3>❌ حدث خطأ</h3>
                <p>{str(e)}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
