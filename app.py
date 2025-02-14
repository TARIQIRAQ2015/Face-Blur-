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
from pdf2image import convert_from_bytes

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

def detect_faces_advanced(image):
    """
    كشف الوجوه باستخدام خوارزميات متعددة مع تحسين الدقة
    """
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # تحميل الكواشف الأساسية
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # كشف الوجوه الأمامية
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # كشف الوجوه الجانبية
        profiles = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # كشف الوجوه الجانبية في الاتجاه المعاكس
        flipped = cv2.flip(gray, 1)
        profiles_flipped = profile_cascade.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # تحويل إحداثيات الوجوه المعكوسة
        if len(profiles_flipped) > 0:
            profiles_flipped = [(gray.shape[1] - x - w, y, w, h) for (x, y, w, h) in profiles_flipped]
        
        # دمج جميع الوجوه المكتشفة
        all_faces = np.array(list(faces) + list(profiles) + list(profiles_flipped))
        
        return all_faces if len(all_faces) > 0 else np.array([])
        
    except Exception as e:
        logger.error(f"خطأ في كشف الوجوه: {str(e)}")
        return np.array([])

def apply_circular_blur(image, face_coordinates, blur_radius=99):
    """
    تطبيق تمويه دائري على الوجه
    """
    try:
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x, y, face_w, face_h = face_coordinates
        center = (x + face_w//2, y + face_h//2)
        radius = int(max(face_w, face_h) * 0.55)  # نصف قطر التمويه
        
        cv2.circle(mask, center, radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (19, 19), 0)
        
        # تمويه المنطقة المحددة
        blurred = cv2.GaussianBlur(img_array, (blur_radius, blur_radius), 30)
        
        # دمج الصورة الأصلية مع التمويه
        mask_3d = np.stack([mask] * 3, axis=2) / 255.0
        result = img_array.copy()
        result = blurred * mask_3d + img_array * (1 - mask_3d)
        
        return result.astype(np.uint8)
        
    except Exception as e:
        logger.error(f"خطأ في تطبيق التمويه: {str(e)}")
        return img_array

def blur_faces_simple(image):
    """
    تمويه الوجوه في الصورة
    """
    try:
        img_array = np.array(image)
        faces = detect_faces_advanced(image)
        
        if len(faces) > 0:
            for face in faces:
                img_array = apply_circular_blur(img_array, face)
            st.success(f"✅ تم العثور على {len(faces)} وجه/وجوه")
        else:
            st.warning("⚠️ لم يتم العثور على وجوه في الصورة")
        
        return Image.fromarray(img_array)
        
    except Exception as e:
        logger.error(f"خطأ في معالجة الصورة: {str(e)}")
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
            dpi=200,
            size=(1200, None),  # تحسين جودة الصورة
            thread_count=4  # تحسين الأداء
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
        info = pdfinfo_from_bytes(pdf_bytes)
        return info['Pages']
    except Exception as e:
        logger.error(f"خطأ في قراءة معلومات PDF: {str(e)}")
        return 0

def process_pdf(pdf_bytes, lang):
    """
    معالجة ملف PDF مع تمويه دائري
    """
    try:
        # التحقق من عدد الصفحات
        images = convert_from_bytes(pdf_bytes, dpi=72)
        total_pages = len(images)
        del images  # تحرير الذاكرة
        
        if total_pages == 0:
            st.error(get_text('no_pages', lang))
            return
            
        if total_pages > 500:
            st.warning(get_text('page_limit_warning', lang))
            total_pages = 500
        
        # إنشاء شريط التقدم
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # قائمة لتخزين الصور المعالجة
        processed_images = []
        
        # معالجة الصفحات في مجموعات
        batch_size = 5
        for batch_start in range(1, total_pages + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_pages)
            progress_text.text(f"{get_text('pdf_processing', lang)} ({batch_start}-{batch_end}/{total_pages})")
            
            for page_num in range(batch_start, batch_end + 1):
                # تحديث شريط التقدم
                progress = (page_num - 1) / total_pages
                progress_bar.progress(progress)
                
                # معالجة الصفحة
                image = process_pdf_page(pdf_bytes, page_num)
                if image:
                    processed = blur_faces_simple(image)
                    processed_images.append(processed)
                    
                    # عرض النتائج
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption=f"{get_text('page', lang)} {page_num} - {get_text('original_image', lang)}", 
                                use_container_width=True)
                    with col2:
                        st.image(processed, caption=f"{get_text('page', lang)} {page_num} - {get_text('processed_image', lang)}", 
                                use_container_width=True)
                
                # تنظيف الذاكرة
                del image
                gc.collect()
        
        progress_bar.progress(1.0)
        progress_text.text(get_text('pdf_complete', lang))
        
        # إنشاء PDF من الصور المعالجة
        if processed_images:
            pdf_output = io.BytesIO()
            processed_images[0].save(
                pdf_output,
                "PDF",
                save_all=True,
                append_images=processed_images[1:],
                resolution=200.0,
                quality=95
            )
            
            # زر تحميل الملف
            st.download_button(
                get_text('download_pdf', lang),
                pdf_output.getvalue(),
                "processed_document.pdf",
                "application/pdf"
            )
            
    except Exception as e:
        logger.error(f"خطأ في معالجة PDF: {str(e)}")
        st.error(get_text('pdf_processing_error', lang))

def load_custom_css():
    """
    تحميل التصميم المخصص
    """
    st.markdown("""
    <style>
    /* التصميم الأساسي */
    .stApp {
        background: linear-gradient(145deg, #0A1128 0%, #1C1E3C 100%);
        color: #E2E8F0;
    }
    
    /* إخفاء العناصر غير المطلوبة */
    #MainMenu, header, footer, .stDeployButton {
        display: none !important;
    }
    
    /* تنسيق العنوان الرئيسي */
    .main-title {
        background: linear-gradient(45deg, #7F00FF, #E100FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
        font-family: 'Tajawal', sans-serif;
        letter-spacing: -1px;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* تنسيق منطقة رفع الملفات */
    .uploadfile-box {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(127, 0, 255, 0.3);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        margin: 2rem 0;
    }
    
    .uploadfile-box:hover {
        border-color: #7F00FF;
        background: rgba(127, 0, 255, 0.05);
        transform: translateY(-2px);
    }
    
    /* تنسيق الأزرار */
    .stButton button {
        background: linear-gradient(45deg, #7F00FF, #E100FF);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(127, 0, 255, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(127, 0, 255, 0.4);
    }
    
    /* تنسيق محدد اللغة */
    .language-selector {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 0.8rem;
        width: 180px;
        margin: 0 auto;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* تنسيق الصور */
    [data-testid="stImage"] {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 40px rgba(127, 0, 255, 0.2);
    }
    
    /* تنسيق التنبيهات */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(127, 0, 255, 0.2);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
        color: #E2E8F0;
        margin: 1rem 0;
    }
    
    /* تأثيرات إضافية */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .gradient-bg {
        background: linear-gradient(-45deg, #7F00FF, #E100FF, #7F00FF);
        background-size: 200% 200%;
        animation: gradient 15s ease infinite;
    }
    
    /* تنسيق شريط التقدم */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7F00FF, #E100FF);
        height: 8px;
        border-radius: 4px;
    }
    </style>
    
    <!-- إضافة الخطوط -->
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;800&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# الترجمات
TRANSLATIONS = {
    'ar': {
        'title': '🎭 أداة تمويه الوجوه',
        'subtitle': 'حماية الخصوصية بتقنيات متقدمة',
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
        'download_pdf': '⬇️ تحميل الملف الكامل بعد المعالجة',
        'page': 'صفحة',
        'pdf_not_supported': 'عذراً، دعم ملفات PDF غير متوفر حالياً',
        'no_pages': 'لم يتم العثور على صفحات في الملف',
        'page_limit_warning': '⚠️ سيتم معالجة أول 500 صفحة فقط',
        'pdf_processing_error': 'حدث خطأ في معالجة الملف',
        'processing_error': 'حدث خطأ أثناء المعالجة',
        'app_error': 'حدث خطأ في التطبيق',
        'loading': 'جاري التحميل...',
        'success': 'تمت العملية بنجاح',
        'error': 'حدث خطأ'
    },
    'en': {
        'title': '🎭 Face Blur Tool',
        'subtitle': 'Advanced Privacy Protection',
        'upload_button': '📤 Upload Image or PDF',
        'upload_help': 'Upload JPG, JPEG, PNG images or PDF files',
        'processing': 'Processing...',
        'original_image': 'Original Image',
        'processed_image': 'Processed Image',
        'download_button': '⬇️ Download Processed Image',
        'no_faces': '⚠️ No faces detected',
        'faces_found': '✅ Found {} face(s)',
        'pdf_processing': '🔄 Processing {} pages...',
        'pdf_complete': '✅ All pages processed!',
        'download_pdf': '⬇️ Download Complete File',
        'page': 'Page',
        'pdf_not_supported': 'PDF support is not available',
        'no_pages': 'No pages found in the file',
        'page_limit_warning': '⚠️ Only first 500 pages will be processed',
        'pdf_processing_error': 'Error processing the file',
        'processing_error': 'Error during processing',
        'app_error': 'Application error occurred',
        'loading': 'Loading...',
        'success': 'Operation completed successfully',
        'error': 'An error occurred'
    }
}

def get_text(key, lang='en'):
    """
    الحصول على النص المترجم
    """
    try:
        return TRANSLATIONS[lang][key]
    except:
        try:
            return TRANSLATIONS['en'][key]
        except:
            logger.error(f"Missing translation for key: {key}")
            return key

def remove_overlapping_faces(faces, overlap_thresh=0.3):
    """
    إزالة التداخلات بين المستطيلات المكتشفة للوجوه
    """
    if len(faces) == 0:
        return []
    
    # تحويل القائمة إلى مصفوفة numpy
    faces = np.array(faces)
    
    # حساب المساحات
    areas = faces[:, 2] * faces[:, 3]
    
    # ترتيب الوجوه حسب المساحة (من الأكبر إلى الأصغر)
    idxs = areas.argsort()[::-1]
    
    # قائمة للاحتفاظ بالوجوه المقبولة
    keep = []
    
    while len(idxs) > 0:
        # إضافة أكبر وجه إلى القائمة
        current_idx = idxs[0]
        keep.append(current_idx)
        
        if len(idxs) == 1:
            break
            
        # حساب نسبة التداخل مع باقي الوجوه
        xx1 = np.maximum(faces[current_idx][0], faces[idxs[1:]][:, 0])
        yy1 = np.maximum(faces[current_idx][1], faces[idxs[1:]][:, 1])
        xx2 = np.minimum(faces[current_idx][0] + faces[current_idx][2],
                        faces[idxs[1:]][:, 0] + faces[idxs[1:]][:, 2])
        yy2 = np.minimum(faces[current_idx][1] + faces[current_idx][3],
                        faces[idxs[1:]][:, 1] + faces[idxs[1:]][:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # حساب المساحة المتداخلة
        overlap = (w * h) / areas[idxs[1:]]
        
        # حذف الوجوه المتداخلة
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return faces[keep].tolist()

def main():
    """
    الدالة الرئيسية للتطبيق
    """
    try:
        # تحميل التصميم
        load_custom_css()
        
        # إعداد الصفحة
        st.set_page_config(
            page_title="Face Blur Tool",
            page_icon="🎭",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # تعيين قيمة افتراضية للغة
        lang = 'en'
        
        try:
            # اختيار اللغة
            col1, col2, col3 = st.columns([3, 1, 3])
            with col2:
                lang = st.selectbox(
                    "🌐",
                    ['ar', 'en'],
                    format_func=lambda x: 'العربية' if x == 'ar' else 'English',
                    label_visibility="collapsed"
                )
        except:
            # في حالة حدوث خطأ، استخدام اللغة الافتراضية
            pass
        
        # العنوان الرئيسي
        st.markdown(f'<div class="main-title">{get_text("title", lang)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="subtitle">{get_text("subtitle", lang)}</div>', unsafe_allow_html=True)
        
        # منطقة رفع الملفات
        uploaded_file = st.file_uploader(
            get_text('upload_button', lang),
            type=["jpg", "jpeg", "png", "pdf"],
            help=get_text('upload_help', lang)
        )
        
        if uploaded_file:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                process_pdf(uploaded_file.getvalue(), lang)
            else:
                try:
                    image = Image.open(uploaded_file)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption=get_text('original_image', lang), use_container_width=True)
                    
                    with st.spinner(get_text('processing', lang)):
                        processed_image = blur_faces_simple(image)
                    
                    with col2:
                        st.image(processed_image, caption=get_text('processed_image', lang), use_container_width=True)
                    
                    # زر التحميل
                    buf = io.BytesIO()
                    processed_image.save(buf, format="PNG", quality=95)
                    st.download_button(
                        get_text('download_button', lang),
                        buf.getvalue(),
                        "blurred_image.png",
                        "image/png"
                    )
                
                except Exception as e:
                    logger.error(f"خطأ في معالجة الصورة: {str(e)}")
                    st.error(get_text('processing_error', lang))
    
    except Exception as e:
        # استخدام اللغة الإنجليزية في حالة الخطأ
        logger.error(f"خطأ في التطبيق: {str(e)}")
        st.error(get_text('app_error', 'en'))

if __name__ == "__main__":
    main()
