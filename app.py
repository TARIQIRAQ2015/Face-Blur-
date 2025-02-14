import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import face_recognition
from pdf2image import convert_from_bytes
from PIL import Image
import io
from typing import List, Tuple, Optional
import logging
import os
import sys
from pathlib import Path
import math
import json
import requests
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
    
    .upload-text {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .success-message {
        padding: 1rem;
        background: #4CAF50;
        color: white;
        border-radius: 10px;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
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

    def create_circular_mask(self, height: int, width: int, center: Tuple[int, int], radius: int) -> np.ndarray:
        """إنشاء قناع دائري للتمويه"""
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
        return mask.astype(np.uint8)
    
    def apply_circular_blur(self, image: np.ndarray, center: Tuple[int, int], radius: int, blur_amount: int = 99) -> np.ndarray:
        """تطبيق تمويه دائري على منطقة محددة"""
        mask = self.create_circular_mask(image.shape[0], image.shape[1], center, radius)
        
        # إنشاء نسخة مموهة من الصورة كاملة
        blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 30)
        
        # دمج الصورة الأصلية مع المنطقة المموهة باستخدام القناع
        result = image.copy()
        result[mask == 1] = blurred[mask == 1]
        
        return result
    
    def get_face_landmarks(self, image: np.ndarray) -> List[dict]:
        """استخراج معالم الوجه باستخدام FaceMesh"""
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return []
        
        landmarks_list = []
        height, width = image.shape[:2]
        
        for face_landmarks in results.multi_face_landmarks:
            # حساب مركز ومحيط الوجه
            x_coords = [landmark.x * width for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * height for landmark in face_landmarks.landmark]
            
            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))
            
            # حساب نصف قطر الدائرة المحيطة بالوجه
            radius = int(max(
                max(x_coords) - min(x_coords),
                max(y_coords) - min(y_coords)
            ) * 0.7)  # 0.7 لتغطية الوجه بشكل أفضل
            
            landmarks_list.append({
                'center': (center_x, center_y),
                'radius': radius
            })
        
        return landmarks_list
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """تحسين جودة الصورة للكشف عن الوجوه الصغيرة"""
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
    
    def detect_small_faces(self, image: np.ndarray) -> List[dict]:
        """كشف الوجوه الصغيرة باستخدام عدة تقنيات"""
        height, width = image.shape[:2]
        faces = []
        
        # تجربة أحجام مختلفة من الصورة
        scales = [1.0, 1.5, 2.0]  # تكبير الصورة للكشف عن الوجوه الصغيرة
        
        for scale in scales:
            # تغيير حجم الصورة
            if scale != 1.0:
                width_scaled = int(width * scale)
                height_scaled = int(height * scale)
                scaled_image = cv2.resize(image, (width_scaled, height_scaled))
            else:
                scaled_image = image
                width_scaled, height_scaled = width, height
            
            # كشف الوجوه باستخدام Haar Cascade
            gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
            cascade_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # تحويل النتائج إلى الحجم الأصلي
            for (x, y, w, h) in cascade_faces:
                center_x = int((x + w/2) / scale)
                center_y = int((y + h/2) / scale)
                radius = int(max(w, h) / scale * 0.7)
                
                faces.append({
                    'center': (center_x, center_y),
                    'radius': radius
                })
        
        return faces
    
    def detect_faces_with_deepface(self, image: np.ndarray) -> List[dict]:
        """استخدام DeepFace للكشف عن الوجوه"""
        try:
            faces = DeepFace.extract_faces(
                image,
                detector_backend='retinaface',
                enforce_detection=False
            )
            return [
                {
                    'center': (
                        int(face['facial_area']['x'] + face['facial_area']['w']/2),
                        int(face['facial_area']['y'] + face['facial_area']['h']/2)
                    ),
                    'radius': int(max(face['facial_area']['w'], face['facial_area']['h']) * 0.7)
                }
                for face in faces
            ]
        except Exception as e:
            logger.warning(f"خطأ في DeepFace: {str(e)}")
            return []

    def detect_faces_with_face_recognition(self, image: np.ndarray) -> List[dict]:
        """استخدام face_recognition للكشف عن الوجوه"""
        try:
            face_locations = face_recognition.face_locations(image, model="cnn")
            return [
                {
                    'center': (
                        int((right + left) / 2),
                        int((bottom + top) / 2)
                    ),
                    'radius': int(max(right - left, bottom - top) * 0.7)
                }
                for top, right, bottom, left in face_locations
            ]
        except Exception as e:
            logger.warning(f"خطأ في face_recognition: {str(e)}")
            return []

    def apply_smart_blur(self, image: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        """تطبيق تمويه ذكي مع تأثيرات متقدمة"""
        mask = self.create_circular_mask(image.shape[0], image.shape[1], center, radius)
        
        # إنشاء تمويه متدرج
        blurred = cv2.GaussianBlur(image, (99, 99), 30)
        
        # إضافة تأثير التدرج للتمويه
        gradient_mask = np.zeros_like(mask, dtype=np.float32)
        for i in range(radius):
            temp_mask = self.create_circular_mask(
                image.shape[0], image.shape[1],
                center, radius - i
            )
            gradient_mask += temp_mask * (1 - i/radius)
        
        gradient_mask = np.clip(gradient_mask, 0, 1)
        
        # دمج الصور
        result = image.copy()
        for c in range(3):  # للقنوات RGB
            result[:,:,c] = (
                image[:,:,c] * (1 - gradient_mask) +
                blurred[:,:,c] * gradient_mask
            )
        
        return result.astype(np.uint8)

    def process_image(self, image: Image.Image) -> Image.Image:
        """معالجة الصورة باستخدام جميع التقنيات المتاحة"""
        try:
            img = np.array(image)
            enhanced_img = self.enhance_image(img)
            
            # جمع الوجوه من جميع الطرق
            all_faces = []
            all_faces.extend(self.get_face_landmarks(enhanced_img))
            all_faces.extend(self.detect_faces_with_deepface(enhanced_img))
            all_faces.extend(self.detect_faces_with_face_recognition(enhanced_img))
            all_faces.extend(self.detect_small_faces(enhanced_img))
            
            # إزالة التكرارات
            unique_faces = self.remove_duplicates(all_faces)
            
            if not unique_faces:
                return image
            
            # تطبيق التمويه الذكي
            for face in unique_faces:
                img = self.apply_smart_blur(img, face['center'], face['radius'])
            
            logger.info(f"تم العثور على {len(unique_faces)} وجه/وجوه")
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
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st_lottie(lottie_face, height=200)
        
        st.title("🎭 أداة تمويه الوجوه الذكية")
        st.markdown("""
        <div class="upload-text">
            <h3>قم برفع صورة أو ملف PDF لتمويه الوجوه تلقائياً</h3>
            <p>نستخدم أحدث تقنيات الذكاء الاصطناعي للكشف عن الوجوه وتمويهها بشكل احترافي</p>
        </div>
        """, unsafe_allow_html=True)
        
        processor = AdvancedFaceBlurProcessor()
        
        # التحقق من تثبيت Poppler
        poppler_installed = check_poppler_installation()
        
        with st.container():
            uploaded_file = st.file_uploader(
                "اختر ملفاً",
                type=["jpg", "jpeg", "png", "pdf"],
                help="يمكنك رفع ملفات بصيغة JPG, JPEG, PNG أو PDF"
            )
        
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.type
                
                if "pdf" in file_type and not poppler_installed:
                    show_poppler_installation_instructions()
                    return
                
                with st.spinner("🔄 جاري معالجة الملف..."):
                    progress_bar = st.progress(0)
                    
                    if "pdf" in file_type:
                        st.info("📄 جاري معالجة ملف PDF...")
                        processed_images = process_pdf(uploaded_file, processor)
                        
                        st.success(f"✅ تم معالجة {len(processed_images)} صفحة/صفحات بنجاح")
                        
                        for idx, img in enumerate(processed_images, 1):
                            progress_bar.progress((idx / len(processed_images)))
                            
                            with st.container():
                                st.markdown(f"### 📄 صفحة {idx}")
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
                        st.markdown("### 📊 معلومات الصورة")
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
                            with st.spinner("🔄 جاري تمويه الوجوه..."):
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
                st.balloons()
            
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء معالجة الملف: {str(e)}")
                logger.error(f"خطأ: {str(e)}")
    
    elif selected == "المعلومات":
        st.title("ℹ️ معلومات عن التطبيق")
        st.markdown("""
        ### 🔍 تقنيات الكشف عن الوجوه
        - **MediaPipe Face Mesh**: للكشف الدقيق عن معالم الوجه
        - **DeepFace**: نموذج متقدم للكشف عن الوجوه
        - **Face Recognition**: مكتبة قوية للتعرف على الوجوه
        - **Haar Cascade**: للكشف عن الوجوه الصغيرة والبعيدة
        
        ### 🛡️ الخصوصية والأمان
        - جميع المعالجة تتم محلياً على جهازك
        - لا يتم حفظ أو مشاركة أي بيانات
        - يمكنك حذف الملفات المعالجة فور تحميلها
        
        ### 📝 المميزات
        - دعم ملفات PDF متعددة الصفحات
        - تمويه ذكي مع تأثيرات متدرجة
        - واجهة مستخدم سهلة وجذابة
        - دعم الصور عالية الدقة
        - كشف الوجوه الصغيرة والبعيدة
        """)

if __name__ == "__main__":
    main()
