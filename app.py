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

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceBlurProcessor:
    def __init__(self, blur_kernel: Tuple[int, int] = (99, 99), blur_sigma: int = 30):
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    
    def blur_faces(self, image: Image.Image) -> Image.Image:
        """
        تطبيق التمويه على الوجوه في الصورة
        
        Args:
            image: صورة PIL
        Returns:
            صورة PIL بعد تمويه الوجوه
        """
        try:
            img = np.array(image)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results = self.face_detection.process(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            
            if not results.detections:
                logger.info("لم يتم العثور على وجوه في الصورة")
                return image
            
            height, width = img.shape[:2]
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * width))
                y = max(0, int(bbox.ymin * height))
                w = min(int(bbox.width * width), width - x)
                h = min(int(bbox.height * height), height - y)
                
                face = img[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(face, self.blur_kernel, self.blur_sigma)
                img[y:y+h, x:x+w] = blurred_face
            
            logger.info(f"تم العثور على {len(results.detections)} وجه/وجوه")
            return Image.fromarray(img)
        
        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {str(e)}")
            raise

def check_poppler_installation() -> bool:
    """التحقق من تثبيت Poppler"""
    try:
        from pdf2image.pdf2image import check_poppler_version
        check_poppler_version()
        return True
    except Exception:
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
    sudo apt-get install -y poppler-utils
    ```
    
    #### على نظام Windows:
    1. قم بتحميل Poppler من [هذا الرابط](https://github.com/oschwartz10612/poppler-windows/releases/)
    2. قم باستخراج الملفات إلى مجلد (مثلاً C:\\Program Files\\poppler)
    3. أضف مسار المجلد bin إلى متغيرات النظام PATH
    
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
