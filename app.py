import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import cv2
import numpy as np
import mediapipe as mp
from pdf2image import convert_from_bytes
from PIL import Image
import io
from typing import List, Tuple, Optional
import logging
import math
import requests
from pathlib import Path
import time

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تحميل الرسوم المتحركة
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# تكوين الصفحة
def set_page_config():
    st.set_page_config(
        page_title="أداة تمويه الوجوه الذكية",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # إضافة CSS مخصص
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main {
        background: rgba(255, 255, 255, 0.95);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #2196F3, #00BCD4);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

class AdvancedFaceBlurProcessor:
    def __init__(self):
        """تهيئة معالج تمويه الوجوه المتقدم"""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=20,
            refine_landmarks=True,
            min_detection_confidence=0.4
        )
        
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.4
        )
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def apply_smart_blur(self, image: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        """تطبيق تمويه ذكي مع تأثيرات متقدمة"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # إنشاء تمويه متدرج
        blurred = cv2.GaussianBlur(image, (99, 99), 30)
        
        # تطبيق تأثير التدرج
        mask_blur = cv2.GaussianBlur(mask, (99, 99), 30)
        mask_blur = mask_blur.astype(float) / 255
        
        # دمج الصور
        result = image.copy()
        for c in range(3):
            result[:,:,c] = (image[:,:,c] * (1 - mask_blur) + 
                           blurred[:,:,c] * mask_blur)
        
        return result.astype(np.uint8)

    def process_image(self, image: Image.Image) -> Image.Image:
        """معالجة الصورة باستخدام جميع التقنيات المتاحة"""
        try:
            img = np.array(image)
            results = self.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not results.detections:
                return image
            
            height, width = img.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                center_x = x + w // 2
                center_y = y + h // 2
                radius = int(max(w, h) * 0.7)
                
                img = self.apply_smart_blur(
                    img,
                    (center_x, center_y),
                    radius
                )
            
            return Image.fromarray(img)
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {str(e)}")
            raise

def check_poppler_installation() -> bool:
    """التحقق من تثبيت Poppler"""
    try:
        # التحقق من وجود الملفات الضرورية
        import shutil
        poppler_path = shutil.which('pdftoppm')
        if poppler_path is None:
            logger.warning("Poppler غير موجود في مسار النظام")
            return False
            
        # محاولة تحويل PDF فارغ للتأكد من عمل المكتبة
        test_pdf = io.BytesIO(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 1 1]>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000015 00000 n\n0000000061 00000 n\n0000000114 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n176\n%%EOF\n")
        convert_from_bytes(test_pdf.getvalue())
        logger.info("تم التحقق من تثبيت Poppler بنجاح")
        return True
    except Exception as e:
        logger.error(f"خطأ في التحقق من Poppler: {str(e)}")
        return False

def process_pdf(pdf_bytes: io.BytesIO, processor: AdvancedFaceBlurProcessor) -> List[Image.Image]:
    """معالجة ملف PDF وتطبيق التمويه على كل صفحة"""
    try:
        if not check_poppler_installation():
            raise RuntimeError(
                "Poppler غير مثبت. يرجى تثبيت Poppler للتعامل مع ملفات PDF."
            )
        
        images = convert_from_bytes(pdf_bytes.read())
        return [processor.process_image(img) for img in images]
    except Exception as e:
        logger.error(f"خطأ في معالجة ملف PDF: {str(e)}")
        raise

def show_header():
    """عرض رأس الصفحة"""
    st.title("🎭 أداة تمويه الوجوه الذكية")
    st.markdown("""
    <div class="upload-text">
        <h3>قم برفع صورة أو ملف PDF لتمويه الوجوه تلقائياً</h3>
        <p>نستخدم أحدث تقنيات الذكاء الاصطناعي للكشف عن الوجوه وتمويهها بشكل احترافي</p>
    </div>
    """, unsafe_allow_html=True)

def show_poppler_installation_instructions():
    """عرض تعليمات تثبيت Poppler"""
    st.error("❌ Poppler غير مثبت")
    st.markdown("""
    ### تعليمات تثبيت Poppler:
    
    #### على نظام Ubuntu/Debian:
    ```bash
    sudo apt-get update
    sudo apt-get install -y poppler-utils libpoppler-dev libpoppler-cpp-dev
    ```
    
    #### على نظام Windows:
    1. قم بتحميل Poppler من [هذا الرابط](https://github.com/oschwartz10612/poppler-windows/releases/)
    2. قم باستخراج الملفات إلى مجلد (مثلاً C:\\Program Files\\poppler)
    3. أضف مسار المجلد bin إلى متغيرات النظام PATH
    4. أعد تشغيل الجهاز
    
    #### على نظام macOS:
    ```bash
    brew install poppler
    ```
    
    بعد التثبيت، قم بإعادة تشغيل التطبيق.
    """)

def main():
    set_page_config()
    
    # تحميل الرسوم المتحركة
    lottie_face = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_UJNc2t.json")
    
    # القائمة الجانبية
    with st.sidebar:
        selected = option_menu(
            "القائمة الرئيسية",
            ["الرئيسية", "المعلومات", "الإعدادات"],
            icons=['house', 'info-circle', 'gear'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "1rem"},
                "icon": {"font-size": "1.2rem"},
                "nav-link": {"font-size": "1rem", "text-align": "right"}
            }
        )
        
        if selected == "الإعدادات":
            st.subheader("⚙️ إعدادات التمويه")
            blur_amount = st.slider(
                "قوة التمويه",
                min_value=1,
                max_value=199,
                value=99,
                step=2
            )
            detection_confidence = st.slider(
                "دقة الكشف عن الوجوه",
                min_value=0.1,
                max_value=1.0,
                value=0.4,
                step=0.1
            )
    
    if selected == "الرئيسية":
        # تصميم الصفحة الرئيسية
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #1E88E5;'>🎭 أداة تمويه الوجوه الذكية</h1>
            <p style='font-size: 1.2rem; color: #424242;'>
                نستخدم أحدث تقنيات الذكاء الاصطناعي للكشف عن الوجوه وتمويهها بشكل احترافي
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st_lottie(lottie_face, height=200, key="face_animation")
        
        processor = AdvancedFaceBlurProcessor()
        
        # التحقق من تثبيت Poppler
        poppler_installed = check_poppler_installation()
        
        with st.container():
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.9); padding: 2rem; border-radius: 15px; margin: 2rem 0;'>
                <h3 style='text-align: center; color: #1E88E5;'>📁 رفع الملفات</h3>
                <p style='text-align: center; color: #424242;'>يمكنك رفع صور بصيغة JPG, JPEG, PNG أو ملفات PDF</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "اختر ملفاً",
                type=["jpg", "jpeg", "png", "pdf"],
                help="اسحب الملف هنا أو انقر لاختيار ملف"
            )
        
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.type
                
                if "pdf" in file_type and not poppler_installed:
                    show_poppler_installation_instructions()
                    return
                
                with st.spinner("🔄 جاري معالجة الملف..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if "pdf" in file_type:
                        status_text.text("📄 جاري معالجة ملف PDF...")
                        processed_images = process_pdf(uploaded_file, processor)
                        
                        st.success(f"✅ تم معالجة {len(processed_images)} صفحة/صفحات بنجاح")
                        
                        for idx, img in enumerate(processed_images, 1):
                            progress_bar.progress(idx / len(processed_images))
                            status_text.text(f"معالجة الصفحة {idx} من {len(processed_images)}")
                            
                            st.markdown(f"""
                            <div style='background: rgba(255, 255, 255, 0.9); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                                <h3 style='color: #1E88E5;'>📄 صفحة {idx}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(img, caption=f"صفحة {idx} بعد المعالجة", use_column_width=True)
                            with col2:
                                buf = io.BytesIO()
                                img.save(buf, format="PNG")
                                st.download_button(
                                    f"⬇️ تحميل الصفحة {idx}",
                                    buf.getvalue(),
                                    f"blurred_page_{idx}.png",
                                    "image/png",
                                    use_container_width=True
                                )
                    else:
                        image = Image.open(uploaded_file)
                        
                        # عرض معلومات الصورة
                        st.markdown("""
                        <div style='background: rgba(255, 255, 255, 0.9); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                            <h3 style='color: #1E88E5;'>📊 معلومات الصورة</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"العرض: {image.width} بكسل")
                        with col2:
                            st.info(f"الارتفاع: {image.height} بكسل")
                        with col3:
                            st.info(f"النوع: {image.mode}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### الصورة الأصلية")
                            st.image(image, caption="الصورة الأصلية", use_column_width=True)
                        
                        with col2:
                            st.markdown("#### الصورة بعد المعالجة")
                            status_text.text("🔄 جاري تمويه الوجوه...")
                            processed_image = processor.process_image(image)
                            st.image(processed_image, caption="الصورة بعد التمويه", use_column_width=True)
                            
                            buf = io.BytesIO()
                            processed_image.save(buf, format="PNG")
                            st.download_button(
                                "⬇️ تحميل الصورة المعدلة",
                                buf.getvalue(),
                                "blurred_image.png",
                                "image/png",
                                use_container_width=True
                            )
                
                progress_bar.progress(100)
                status_text.empty()
                st.balloons()
                
                st.success("✨ تمت المعالجة بنجاح!")
            
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء معالجة الملف: {str(e)}")
                logger.error(f"خطأ: {str(e)}")
    
    elif selected == "المعلومات":
        st.title("ℹ️ معلومات عن التطبيق")
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.9); padding: 2rem; border-radius: 15px;'>
            <h3 style='color: #1E88E5;'>🔍 المميزات</h3>
            <ul>
                <li>كشف الوجوه باستخدام تقنيات الذكاء الاصطناعي المتقدمة</li>
                <li>تمويه ذكي مع تأثيرات متدرجة</li>
                <li>دعم الصور عالية الدقة</li>
                <li>معالجة ملفات PDF متعددة الصفحات</li>
                <li>واجهة مستخدم سهلة وجذابة</li>
            </ul>
            
            <h3 style='color: #1E88E5;'>🛡️ الخصوصية والأمان</h3>
            <ul>
                <li>جميع المعالجة تتم محلياً على جهازك</li>
                <li>لا يتم حفظ أو مشاركة أي بيانات</li>
                <li>يمكنك حذف الملفات المعالجة فور تحميلها</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
