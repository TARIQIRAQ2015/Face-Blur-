import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import io
import logging
import subprocess
import sys
import os
import time
import gc
import urllib.request
import bz2

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

def detect_and_blur_face_advanced(image):
    """
    كشف وتمويه الوجوه بشكل دقيق
    """
    try:
        # تحويل الصورة إلى مصفوفة numpy
        img_array = np.array(image)
        
        # كشف مواقع الوجوه باستخدام OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            st.warning("⚠️ لم يتم العثور على وجوه في الصورة")
            return image
        
        # تمويه كل وجه
        for (x, y, w, h) in faces:
            # توسيع منطقة الوجه قليلاً
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_array.shape[1], x + w + padding)
            y2 = min(img_array.shape[0], y + h + padding)
            
            # تمويه منطقة الوجه
            face_roi = img_array[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            img_array[y1:y2, x1:x2] = blurred_face
        
        st.success(f"✅ تم العثور على {len(faces)} وجه/وجوه")
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
                    processed_image = detect_and_blur_face_advanced(image)
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

def load_css():
    st.markdown("""
    <style>
    /* إخفاء العناصر غير المرغوبة */
    #MainMenu, header, footer, .stDeployButton, [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* الخلفية الرئيسية */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        margin-top: -4rem;
    }
    
    /* تنسيق العنوان الرئيسي */
    .main-title {
        font-family: 'Tajawal', 'Roboto', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 2rem;
        margin: 2rem 0;
        background: linear-gradient(45deg, #3B82F6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
    }
    
    /* تنسيق منطقة رفع الملفات */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(59, 130, 246, 0.3);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3B82F6;
        background: rgba(59, 130, 246, 0.1);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
    }
    
    /* تنسيق الأزرار */
    .stButton button {
        background: linear-gradient(45deg, #3B82F6, #60A5FA);
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* تنسيق الصور */
    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
    }
    
    /* تنسيق التنبيهات */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        color: white !important;
    }
    
    /* تنسيق النصوص */
    .stMarkdown {
        color: #E2E8F0;
    }
    
    h1, h2, h3 {
        color: #F8FAFC;
    }
    
    /* تنسيق محدد اللغة */
    .stSelectbox [data-testid="stMarkdown"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* تنسيق شريط التقدم */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3B82F6, #60A5FA);
        height: 8px;
        border-radius: 4px;
    }
    
    /* تنسيق الأعمدة */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* تأثيرات إضافية */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 80% 80%, rgba(96, 165, 250, 0.15) 0%, transparent 40%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* تنسيق النص باللغة العربية */
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
        line-height: 1.8;
        color: #E2E8F0;
    }
    
    /* تنسيق النص باللغة الإنجليزية */
    .english-text {
        direction: ltr;
        text-align: left;
        font-family: 'Roboto', sans-serif;
        line-height: 1.8;
        color: #E2E8F0;
    }
    
    /* تنسيق الفواصل */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(59, 130, 246, 0.3) 50%, 
            transparent 100%
        );
        margin: 2rem 0;
    }
    </style>
    
    <!-- إضافة الخطوط -->
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# إضافة الترجمات
TRANSLATIONS = {
    'ar': {
        'title': '🎭 أداة تمويه الوجوه',
        'upload_button': '📤 ارفع صورة أو ملف PDF',
        'upload_help': 'يمكنك رفع صور بصيغ JPG, JPEG, PNG أو ملف PDF',
        'processing': 'جاري المعالجة...',
        'original_image': 'الصورة الأصلية',
        'processed_image': 'الصورة بعد التمويه',
        'download_button': '⬇️ تحميل الصورة المعدلة',
        'no_faces': '⚠️ لم يتم العثور على وجوه في الصورة',
        'faces_found': '✅ تم العثور على {} وجه/وجوه',
        'pdf_processing': '🔄 جاري معالجة {} صفحة...',
        'pdf_complete': '✅ تمت معالجة جميع الصفحات!',
        'download_pdf': '⬇️ تحميل الملف الكامل بعد المعالجة (PDF)',
        'notes': '📝 ملاحظات',
        'note_formats': 'يمكنك رفع صور بصيغ JPG, JPEG, PNG أو ملف PDF',
        'note_pdf': 'معالجة ملفات PDF قد تستغرق بعض الوقت حسب عدد الصفحات',
    },
    'en': {
        'title': '🎭 Face Blur Tool',
        'upload_button': '📤 Upload Image or PDF',
        'upload_help': 'You can upload JPG, JPEG, PNG images or PDF files',
        'processing': 'Processing...',
        'original_image': 'Original Image',
        'processed_image': 'Processed Image',
        'download_button': '⬇️ Download Processed Image',
        'no_faces': '⚠️ No faces detected in the image',
        'faces_found': '✅ Found {} face(s)',
        'pdf_processing': '🔄 Processing {} pages...',
        'pdf_complete': '✅ All pages processed!',
        'download_pdf': '⬇️ Download Complete Processed File (PDF)',
        'notes': '📝 Notes',
        'note_formats': 'You can upload JPG, JPEG, PNG images or PDF files',
        'note_pdf': 'Processing PDF files may take some time depending on the number of pages',
    }
}

def get_text(key, lang, *args):
    """
    الحصول على النص المترجم
    """
    text = TRANSLATIONS[lang][key]
    if args:
        text = text.format(*args)
    return text

def main():
    try:
        load_css()
        configure_page()
        
        # العنوان الرئيسي والترجمة في صف واحد
        st.markdown('<div class="main-title">🎭 أداة تمويه الوجوه / Face Blur Tool</div>', unsafe_allow_html=True)
        
        # محدد اللغة في وسط الصفحة
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            lang = st.selectbox(
                "🌐",
                ['ar', 'en'],
                format_func=lambda x: 'العربية' if x == 'ar' else 'English',
                label_visibility="collapsed",
                key="language-selector"
            )
        
        st.markdown("---")
        
        # منطقة رفع الملفات
        uploaded_file = st.file_uploader(
            get_text('upload_button', lang),
            type=["jpg", "jpeg", "png", "pdf"],
            help=get_text('upload_help', lang)
        )
        
        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                if file_extension == 'pdf':
                    if not PDF_SUPPORT:
                        st.error(get_text('pdf_not_available', lang))
                        return
                    
                    with st.spinner(get_text('processing', lang)):
                        process_pdf(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption=get_text("original_image", lang), use_container_width=True)
                    
                    with st.spinner(get_text('processing', lang)):
                        processed_image = detect_and_blur_face_advanced(image)
                    
                    with col2:
                        st.image(processed_image, caption=get_text("processed_image", lang), use_container_width=True)
                    
                    # زر التحميل
                    buf = io.BytesIO()
                    processed_image.save(buf, format="PNG")
                    st.download_button(
                        get_text('download_button', lang),
                        buf.getvalue(),
                        "blurred_image.png",
                        "image/png"
                    )
            
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                st.error(get_text('processing_error', lang))
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(get_text('app_error', lang))

if __name__ == "__main__":
    main()
