import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import logging
import subprocess
import sys
import os
import time
import gc

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_poppler():
    """
    التحقق من تثبيت Poppler وإظهار معلومات التثبيت
    """
    try:
        # محاولة تنفيذ أمر pdftoppm للتحقق من وجود Poppler
        result = subprocess.run(['pdftoppm', '-v'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        
        if result.stderr:
            logger.info(f"Poppler version: {result.stderr.strip()}")
            return True
        return False
    except FileNotFoundError:
        logger.error("Poppler not found in system PATH")
        # محاولة البحث عن مسار Poppler
        possible_paths = [
            '/usr/bin/pdftoppm',
            '/usr/local/bin/pdftoppm',
            '/opt/homebrew/bin/pdftoppm'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found Poppler at: {path}")
                os.environ['PATH'] = f"{os.path.dirname(path)}:{os.environ.get('PATH', '')}"
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking Poppler: {str(e)}")
        return False

# التحقق من وجود مكتبة pdf2image و Poppler
PDF_SUPPORT = False
try:
    from pdf2image import convert_from_bytes
    if check_poppler():
        PDF_SUPPORT = True
        logger.info("PDF support enabled")
    else:
        logger.warning("Poppler غير مثبت في النظام")
        st.warning("""
        لتمكين دعم ملفات PDF، يجب تثبيت Poppler:
        
        على Linux:
        ```bash
        sudo apt-get update
        sudo apt-get install -y poppler-utils
        ```
        
        على macOS:
        ```bash
        brew install poppler
        ```
        
        على Windows:
        قم بتحميل وتثبيت Poppler من:
        http://blog.alivate.com.au/poppler-windows/
        """)
except ImportError as e:
    logger.warning(f"مكتبة pdf2image غير متوفرة: {str(e)}")

def configure_page():
    try:
        st.set_page_config(
            page_title="أداة تمويه الوجوه",
            page_icon="👤",
            layout="wide"
        )
    except Exception as e:
        logger.error(f"خطأ في تهيئة الصفحة: {str(e)}")

def blur_faces_simple(image):
    """
    تمويه الوجوه بشكل دائري يتناسب مع حجم الوجه
    """
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # تحميل كواشف الوجوه المختلفة
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # كشف الوجوه الأمامية
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),  # حجم أصغر للوجوه
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # كشف الوجوه الجانبية
        profiles = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # دمج جميع الوجوه المكتشفة
        all_faces = list(faces) + list(profiles)
        
        # تمويه كل وجه
        for (x, y, w, h) in all_faces:
            # إنشاء قناع دائري
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w//2, h//2)
            radius = min(w, h)//2
            cv2.circle(mask, center, radius, 255, -1)
            
            # تمويه منطقة الوجه
            face_roi = img_array[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            
            # تطبيق القناع الدائري
            mask_3d = np.stack([mask]*3, axis=2) / 255.0
            face_roi[:] = blurred_face * mask_3d + face_roi * (1 - mask_3d)
        
        if not all_faces:
            st.warning("⚠️ لم يتم العثور على وجوه في الصورة")
            
        return Image.fromarray(img_array)
    except Exception as e:
        logger.error(f"خطأ في معالجة الصورة: {str(e)}")
        st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
        return image

@st.cache_data
def process_pdf_page(pdf_bytes, page_number):
    """
    معالجة صفحة واحدة من ملف PDF
    """
    try:
        images = convert_from_bytes(
            pdf_bytes,
            first_page=page_number,
            last_page=page_number,
            dpi=150,  # تقليل الدقة أكثر للملفات الكبيرة
            size=(800, None),  # تقليل حجم الصورة
            thread_count=2  # استخدام المعالجة المتوازية
        )
        return images[0] if images else None
    except Exception as e:
        logger.error(f"خطأ في معالجة صفحة PDF: {str(e)}")
        return None

def get_pdf_page_count(pdf_bytes):
    """
    الحصول على عدد صفحات ملف PDF
    """
    try:
        from pdf2image.pdf2image import pdfinfo_from_bytes
        info = pdfinfo_from_bytes(pdf_bytes)
        return info['Pages']
    except Exception as e:
        logger.error(f"خطأ في قراءة معلومات PDF: {str(e)}")
        return 0

def process_pdf(pdf_bytes):
    """
    معالجة ملف PDF وتحويله إلى صور
    """
    if not PDF_SUPPORT:
        st.error("عذراً، دعم ملفات PDF غير متوفر حالياً")
        return []
        
    try:
        total_pages = get_pdf_page_count(pdf_bytes.getvalue())
        
        if total_pages == 0:
            st.error("لم يتم العثور على صفحات في ملف PDF")
            return []
            
        if total_pages > 500:
            st.warning("⚠️ يمكن معالجة 500 صفحة كحد أقصى. سيتم معالجة أول 500 صفحة فقط.")
            total_pages = 500
        
        st.info(f"🔄 جاري معالجة {total_pages} صفحة...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # قائمة لتخزين جميع الصور المعالجة
        all_processed_images = []
        
        batch_size = 10
        for batch_start in range(1, total_pages + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_pages)
            status_text.text(f"معالجة الصفحات {batch_start} إلى {batch_end}...")
            
            for page_num in range(batch_start, batch_end + 1):
                progress_bar.progress((page_num - 1) / total_pages)
                
                image = process_pdf_page(pdf_bytes.getvalue(), page_num)
                if image:
                    processed_image = blur_faces_simple(image)
                    all_processed_images.append(processed_image)
                    
                    # عرض الصور
                    st.markdown(f"### صفحة {page_num}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="الصورة الأصلية", use_container_width=True)
                    with col2:
                        st.image(processed_image, caption="الصورة بعد التمويه", use_container_width=True)
                
                del image
            
            gc.collect()
        
        progress_bar.progress(1.0)
        status_text.text("✅ تمت معالجة جميع الصفحات!")
        
        # إنشاء ملف PDF يحتوي على جميع الصور المعالجة
        if all_processed_images:
            pdf_output = io.BytesIO()
            all_processed_images[0].save(
                pdf_output,
                "PDF",
                save_all=True,
                append_images=all_processed_images[1:],
                resolution=150.0,
                quality=85
            )
            
            st.download_button(
                "⬇️ تحميل الملف الكامل بعد المعالجة (PDF)",
                pdf_output.getvalue(),
                "processed_document.pdf",
                "application/pdf"
            )
        
        return []
        
    except Exception as e:
        logger.error(f"خطأ في معالجة ملف PDF: {str(e)}")
        st.error(f"حدث خطأ في معالجة ملف PDF: {str(e)}")
        return []

def main():
    try:
        configure_page()
        
        st.title("🎭 أداة تمويه الوجوه")
        st.markdown("---")
        
        if not PDF_SUPPORT:
            st.warning("""
            ### ⚠️ دعم ملفات PDF غير متوفر
            لتمكين دعم ملفات PDF، تأكد من تثبيت المكتبات المطلوبة.
            """)
        
        # إعدادات التمويه
        blur_intensity = st.slider("شدة التمويه", 25, 199, 99, step=2)
        
        # رفع الملف
        uploaded_file = st.file_uploader(
            "📤 ارفع صورة أو ملف PDF",
            type=["jpg", "jpeg", "png", "pdf"],
            help="يمكنك رفع صور بصيغ JPG, JPEG, PNG أو ملف PDF"
        )
        
        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                if file_extension == 'pdf':
                    if not PDF_SUPPORT:
                        st.error("عذراً، دعم ملفات PDF غير متوفر حالياً.")
                        return
                    
                    with st.spinner("جاري معالجة ملف PDF..."):
                        process_pdf(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption="الصورة الأصلية", use_container_width=True)
                    
                    with st.spinner("جاري معالجة الصورة..."):
                        processed_image = blur_faces_simple(image)
                    
                    with col2:
                        st.image(processed_image, caption="الصورة بعد التمويه", use_container_width=True)
                    
                    # تحميل الصورة المعالجة
                    buf = io.BytesIO()
                    processed_image.save(buf, format="PNG")
                    st.download_button(
                        "⬇️ تحميل الصورة المعدلة",
                        buf.getvalue(),
                        "blurred_image.png",
                        "image/png"
                    )
            
            except Exception as e:
                logger.error(f"خطأ في معالجة الملف: {str(e)}")
                st.error(f"حدث خطأ في معالجة الملف: {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        ### 📝 ملاحظات:
        - يمكنك رفع صور بصيغ JPG, JPEG, PNG""" + 
        (" أو ملف PDF" if PDF_SUPPORT else "") + """
        - استخدم شريط التمرير للتحكم في شدة التمويه
        """ + ("""
        - معالجة ملفات PDF قد تستغرق بعض الوقت حسب عدد الصفحات""" if PDF_SUPPORT else ""))
        
    except Exception as e:
        logger.error(f"خطأ في التطبيق: {str(e)}")
        st.error(f"حدث خطأ في التطبيق: {str(e)}")

if __name__ == "__main__":
    main()
