import streamlit as st
import cv2
import numpy as np
import face_recognition
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

class AdvancedFaceDetector:
    def detect_faces(self, image: np.ndarray) -> list:
        faces = []
        # تحويل الصورة إلى RGB لمكتبة face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # كشف الوجوه بأحجام مختلفة
        face_locations = face_recognition.face_locations(
            rgb_image,
            number_of_times_to_upsample=2,  # زيادة للكشف عن الوجوه الصغيرة
            model="cnn"  # استخدام نموذج CNN للدقة العالية
        )
        
        for face_location in face_locations:
            top, right, bottom, left = face_location
            faces.append({
                'box': [left, top, right - left, bottom - top],
                'confidence': 1.0
            })
        
        return faces

class FaceBlurProcessor:
    def __init__(self):
        self.detector = AdvancedFaceDetector()
    
    def apply_strong_blur(self, image: np.ndarray, box: list) -> np.ndarray:
        x, y, w, h = box
        
        # توسيع منطقة التمويه
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(w + 2*padding_x, image.shape[1] - x)
        h = min(h + 2*padding_y, image.shape[0] - y)
        
        center = (x + w//2, y + h//2)
        radius = int(max(w, h) * 0.9)
        
        # إنشاء قناع متدرج للتمويه
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.circle(mask, center, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (99, 99), 30)
        
        # تطبيق تمويه قوي متعدد المستويات
        blurred = cv2.GaussianBlur(image, (99, 99), 30)
        blurred = cv2.GaussianBlur(blurred, (99, 99), 30)  # تمويه مضاعف
        
        # دمج الصور
        mask = np.expand_dims(mask, -1)
        result = image * (1 - mask) + blurred * mask
        
        return result.astype(np.uint8)
    
    def process_image(self, image: Image.Image) -> tuple:
        try:
            img = np.array(image)
            
            # تحسين جودة الصورة
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # كشف الوجوه
            faces = self.detector.detect_faces(img)
            
            if faces:
                for face in faces:
                    img = self.apply_strong_blur(img, face['box'])
            
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
