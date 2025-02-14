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

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# التحقق من وجود Poppler
def check_poppler():
    try:
        # محاولة تنفيذ أمر pdftoppm للتحقق من وجود Poppler
        subprocess.run(['pdftoppm', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

# التحقق من وجود مكتبة pdf2image و Poppler
PDF_SUPPORT = False
try:
    from pdf2image import convert_from_bytes
    if check_poppler():
        PDF_SUPPORT = True
    else:
        logger.warning("Poppler غير مثبت في النظام")
except ImportError:
    logger.warning("مكتبة pdf2image غير متوفرة")

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
    نسخة مبسطة من تمويه الوجوه باستخدام كاشف الوجوه المدمج في OpenCV
    """
    try:
        # تحويل الصورة إلى مصفوفة numpy
        img_array = np.array(image)
        
        # تحويل الصورة إلى رمادي
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # تحميل كاشف الوجوه
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # كشف الوجوه
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # تمويه كل وجه
        for (x, y, w, h) in faces:
            face = img_array[y:y+h, x:x+w]
            face = cv2.GaussianBlur(face, (99, 99), 30)
            img_array[y:y+h, x:x+w] = face
            
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
        # الحصول على عدد الصفحات
        total_pages = get_pdf_page_count(pdf_bytes.getvalue())
        
        if total_pages == 0:
            st.error("لم يتم العثور على صفحات في ملف PDF")
            return []
            
        if total_pages > 500:  # زيادة الحد الأقصى إلى 500 صفحة
            st.warning("⚠️ يمكن معالجة 500 صفحة كحد أقصى. سيتم معالجة أول 500 صفحة فقط.")
            total_pages = 500
        
        st.info(f"🔄 جاري معالجة {total_pages} صفحة...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_images = []
        batch_size = 10  # معالجة الصفحات في مجموعات
        
        for batch_start in range(1, total_pages + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_pages)
            status_text.text(f"معالجة الصفحات {batch_start} إلى {batch_end}...")
            
            for page_num in range(batch_start, batch_end + 1):
                # تحديث شريط التقدم
                progress_bar.progress((page_num - 1) / total_pages)
                
                # معالجة الصفحة
                image = process_pdf_page(pdf_bytes.getvalue(), page_num)
                if image:
                    processed_images.append(image)
                    
                    # عرض الصورة مباشرة بعد معالجتها
                    st.markdown(f"### صفحة {page_num}")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption="الصورة الأصلية", use_column_width=True)
                    
                    processed_image = blur_faces_simple(image)
                    
                    with col2:
                        st.image(processed_image, caption="الصورة بعد التمويه", use_column_width=True)
                    
                    # زر التحميل
                    buf = io.BytesIO()
                    processed_image.save(buf, format="PNG")
                    st.download_button(
                        f"⬇️ تحميل الصفحة {page_num}",
                        buf.getvalue(),
                        f"blurred_page_{page_num}.png",
                        "image/png"
                    )
                
                # تحرير الذاكرة
                del image
                
            # تحرير الذاكرة بعد كل مجموعة
            import gc
            gc.collect()
        
        progress_bar.progress(1.0)
        status_text.text("✅ تمت معالجة جميع الصفحات!")
        return []  # لا نحتاج لإرجاع الصور لأننا نعرضها مباشرة
        
    except Exception as e:
        logger.error(f"خطأ في معالجة ملف PDF: {str(e)}")
        st.error(f"حدث خطأ في معالجة ملف PDF: {str(e)}")
        return []

def main():
    try:
        configure_page()
        
        st.title("🎭 أداة تمويه الوجوه")
        st.markdown("---")
        
        # إعدادات التمويه
        blur_intensity = st.slider("شدة التمويه", 25, 199, 99, step=2)
        
        # تحديد أنواع الملفات المدعومة
        allowed_types = ["jpg", "jpeg", "png", "pdf"]  # إضافة PDF مباشرة
        
        # رفع الملف
        uploaded_file = st.file_uploader(
            "📤 ارفع صورة أو ملف PDF",
            type=allowed_types,
            help="يمكنك رفع صور بصيغ JPG, JPEG, PNG أو ملف PDF"
        )
        
        if uploaded_file is not None:
            try:
                # التحقق من نوع الملف بناءً على الامتداد
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                if file_extension == 'pdf':
                    if not PDF_SUPPORT:
                        st.error("عذراً، دعم ملفات PDF غير متوفر حالياً. الرجاء تأكد من تثبيت المكتبات المطلوبة.")
                        return
                        
                    with st.spinner("جاري معالجة ملف PDF..."):
                        process_pdf(uploaded_file)
                else:
                    # معالجة الصور العادية
                    image = Image.open(uploaded_file)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption="الصورة الأصلية")
                    
                    with st.spinner("جاري معالجة الصورة..."):
                        processed_image = blur_faces_simple(image)
                    
                    with col2:
                        st.image(processed_image, caption="الصورة بعد التمويه")
                    
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
