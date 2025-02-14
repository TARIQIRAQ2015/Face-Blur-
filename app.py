import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from pdf2image import convert_from_bytes
from PIL import Image
import io
from typing import List, Tuple, Optional
import logging
import os
import sys
from pathlib import Path
import math

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceBlurProcessor:
    def __init__(self):
        """تهيئة معالج تمويه الوجوه مع نماذج متقدمة"""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
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
    
    def blur_faces(self, image: Image.Image) -> Image.Image:
        """
        تطبيق التمويه الدائري على الوجوه في الصورة
        
        Args:
            image: صورة PIL
        Returns:
            صورة PIL بعد تمويه الوجوه
        """
        try:
            # تحويل الصورة إلى مصفوفة NumPy
            img = np.array(image)
            
            # كشف معالم الوجوه
            face_landmarks = self.get_face_landmarks(img)
            
            if not face_landmarks:
                # استخدام نموذج الكشف عن الوجوه كاحتياطي
                results = self.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.detections:
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
                        
                        face_landmarks.append({
                            'center': (center_x, center_y),
                            'radius': radius
                        })
            
            if not face_landmarks:
                logger.info("لم يتم العثور على وجوه في الصورة")
                return image
            
            # تطبيق التمويه الدائري على كل وجه
            for face in face_landmarks:
                img = self.apply_circular_blur(
                    img,
                    face['center'],
                    face['radius']
                )
            
            logger.info(f"تم العثور على {len(face_landmarks)} وجه/وجوه")
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

def process_pdf(pdf_bytes: io.BytesIO, processor: FaceBlurProcessor) -> List[Image.Image]:
    """معالجة ملف PDF وتطبيق التمويه على كل صفحة"""
    try:
        if not check_poppler_installation():
            raise RuntimeError(
                "Poppler غير مثبت. يرجى تثبيت Poppler للتعامل مع ملفات PDF."
            )
        
        images = convert_from_bytes(pdf_bytes.read())
        return [processor.blur_faces(img) for img in images]
    except Exception as e:
        logger.error(f"خطأ في معالجة ملف PDF: {str(e)}")
        raise

def set_page_config():
    """إعداد تكوين الصفحة"""
    st.set_page_config(
        page_title="أداة تمويه الوجوه",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def show_header():
    """عرض رأس الصفحة"""
    st.title("🎭 أداة تمويه الوجوه باستخدام الذكاء الاصطناعي")
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
    }
    </style>
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
    show_header()
    
    processor = FaceBlurProcessor()
    
    # التحقق من تثبيت Poppler
    poppler_installed = check_poppler_installation()
    
    with st.container():
        st.markdown("""
        <div class="upload-text">
            <h3>قم برفع صورة أو ملف PDF لتمويه الوجوه تلقائياً</h3>
            <p>يمكنك رفع الملفات بصيغة JPG, JPEG, PNG أو PDF</p>
        </div>
        """, unsafe_allow_html=True)
        
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
            
            with st.spinner("جاري معالجة الملف..."):
                if "pdf" in file_type:
                    st.info("📄 جاري معالجة ملف PDF...")
                    processed_images = process_pdf(uploaded_file, processor)
                    
                    st.success(f"✅ تم معالجة {len(processed_images)} صفحة/صفحات بنجاح")
                    
                    for idx, img in enumerate(processed_images, 1):
                        with st.container():
                            st.markdown(f"### صفحة {idx}")
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
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="الصورة الأصلية", use_column_width=True)
                    
                    with col2:
                        processed_image = processor.blur_faces(image)
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
        
        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء معالجة الملف: {str(e)}")
            logger.error(f"خطأ: {str(e)}")
    
    # إضافة معلومات إضافية في نهاية الصفحة
    with st.expander("ℹ️ معلومات عن الأداة"):
        st.markdown("""
        - تستخدم هذه الأداة تقنيات الذكاء الاصطناعي للكشف عن الوجوه وتمويهها تلقائياً
        - يمكنك معالجة الصور بصيغ JPG, JPEG, PNG
        - يمكنك أيضاً معالجة ملفات PDF (يتطلب تثبيت Poppler)
        - جميع المعالجة تتم محلياً على جهازك
        - البيانات لا يتم حفظها أو مشاركتها مع أي طرف خارجي
        """)

if __name__ == "__main__":
    main()
