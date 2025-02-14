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
from deepface import DeepFace
import face_recognition
from retinaface import RetinaFace
from mtcnn import MTCNN
import insightface
import torch

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
        page_title="تمويه الوجوه الذكي",
        page_icon="🎭",
        layout="wide"
    )
    
    # تصميم CSS المحسن
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    
    :root {
        --primary-color: #2962FF;
        --secondary-color: #0039CB;
        --background-color: #F5F7FA;
        --text-color: #1A237E;
        --card-background: rgba(255, 255, 255, 0.95);
    }
    
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
    
    .title-container {
        text-align: center;
        padding: 2rem;
        margin-bottom: 2rem;
        background: var(--card-background);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
    }
    
    .title-container h1 {
        color: var(--primary-color);
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .upload-container {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        width: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(41, 98, 255, 0.3);
    }
    
    .results-container {
        background: var(--card-background);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        margin-top: 2rem;
    }
    
    .info-text {
        color: var(--text-color);
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .progress-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    .image-preview {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    </style>
    """, unsafe_allow_html=True)

class AdvancedFaceDetector:
    def __init__(self):
        """تهيئة جميع نماذج كشف الوجوه"""
        self.retina = RetinaFace.build_model()
        self.mtcnn = MTCNN()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=20,
            refine_landmarks=True,
            min_detection_confidence=0.4
        )
        # تهيئة InsightFace
        self.insight_model = insightface.app.FaceAnalysis()
        self.insight_model.prepare(ctx_id=0)
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """كشف الوجوه باستخدام عدة تقنيات متقدمة"""
        faces = []
        height, width = image.shape[:2]
        
        try:
            # 1. RetinaFace
            retina_faces = RetinaFace.detect_faces(image)
            if isinstance(retina_faces, dict):
                for face in retina_faces.values():
                    box = face['facial_area']
                    faces.append({
                        'box': box,
                        'confidence': face['score'],
                        'source': 'retinaface'
                    })
        except Exception as e:
            logger.warning(f"خطأ في RetinaFace: {str(e)}")
        
        try:
            # 2. MTCNN
            mtcnn_faces = self.mtcnn.detect_faces(image)
            for face in mtcnn_faces:
                if face['confidence'] > 0.9:
                    faces.append({
                        'box': face['box'],
                        'confidence': face['confidence'],
                        'source': 'mtcnn'
                    })
        except Exception as e:
            logger.warning(f"خطأ في MTCNN: {str(e)}")
        
        try:
            # 3. InsightFace
            insight_faces = self.insight_model.get(image)
            for face in insight_faces:
                box = face.bbox.astype(int)
                faces.append({
                    'box': [box[0], box[1], box[2]-box[0], box[3]-box[1]],
                    'confidence': face.det_score,
                    'source': 'insightface'
                })
        except Exception as e:
            logger.warning(f"خطأ في InsightFace: {str(e)}")
        
        # إزالة التكرارات وتحسين النتائج
        return self.merge_detections(faces)
    
    def merge_detections(self, faces: List[dict]) -> List[dict]:
        """دمج وتنقية نتائج الكشف"""
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
                # حساب نسبة التداخل
                iou = self.calculate_iou(box1, box2)
                if iou > 0.5:  # إذا كان التداخل كبيراً
                    should_add = False
                    break
            
            if should_add:
                final_faces.append(face)
        
        return final_faces
    
    @staticmethod
    def calculate_iou(box1, box2):
        """حساب نسبة التداخل بين مربعين"""
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

class AdvancedFaceBlurProcessor:
    def __init__(self):
        self.detector = AdvancedFaceDetector()
    
    def apply_smart_blur(self, image: np.ndarray, box: List[int], confidence: float) -> np.ndarray:
        """تطبيق تمويه ذكي مع تأثيرات متدرجة"""
        x, y, w, h = box
        center = (x + w//2, y + h//2)
        radius = int(max(w, h) * 0.6)  # تقليل حجم منطقة التمويه قليلاً
        
        # إنشاء قناع دائري متدرج
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        for r in range(radius):
            cv2.circle(
                mask,
                center,
                radius - r,
                color=(1 - r/radius),
                thickness=1
            )
        
        # تطبيق التمويه بقوة متناسبة مع مستوى الثقة
        blur_strength = int(99 * confidence) | 1  # يجب أن يكون رقماً فردياً
        blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
        
        # دمج الصور
        mask = np.expand_dims(mask, -1)
        result = image * (1 - mask) + blurred * mask
        
        return result.astype(np.uint8)
    
    def process_image(self, image: Image.Image) -> Image.Image:
        """معالجة الصورة وتمويه الوجوه"""
        try:
            # تحويل الصورة إلى مصفوفة NumPy
            img = np.array(image)
            
            # كشف الوجوه
            faces = self.detector.detect_faces(img)
            
            if not faces:
                logger.info("لم يتم العثور على وجوه في الصورة")
                return image
            
            # تطبيق التمويه على كل وجه
            for face in faces:
                img = self.apply_smart_blur(
                    img,
                    face['box'],
                    face['confidence']
                )
            
            logger.info(f"تم العثور على {len(faces)} وجه/وجوه")
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
