import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter
import io
import logging
import subprocess
import sys
import os
import time
import gc
import mediapipe as mp
from PIL import ImageDraw

# تكوين الصفحة - يجب أن يكون أول أمر Streamlit
st.set_page_config(
    page_title="Face Blur Tool | أداة تمويه الوجوه",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# التحقق من دعم PDF
try:
    result = subprocess.run(['pdftoppm', '-v'], capture_output=True, text=True)
    logger.info(f"Poppler version: {result.stderr}")
    PDF_SUPPORT = True
except FileNotFoundError:
    logger.warning("Poppler not found. PDF support disabled.")
    PDF_SUPPORT = False

# إعداد MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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

def detect_and_blur_faces(image):
    """
    كشف وتمويه الوجوه بشكل دائري نقي
    """
    try:
        # تحويل الصورة إلى مصفوفة numpy
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # نسخة للنتيجة النهائية
        result = img_array.copy()
        
        # تحسين الصورة للكشف
        enhanced = cv2.convertScaleAbs(img_array, alpha=1.2, beta=15)
        
        # كشف الوجوه باستخدام MediaPipe
        with mp_face_detection.FaceDetection(
            model_selection=1,  # نموذج كامل للدقة العالية
            min_detection_confidence=0.75  # دقة عالية مع مرونة معقولة
        ) as face_detector:
            # محاولة الكشف على الصورة الأصلية والمحسنة
            results = face_detector.process(img_array)
            if not results.detections:
                results = face_detector.process(enhanced)
                if not results.detections:
                    st.warning(TRANSLATIONS['ar']['no_faces'])
                    return image

            faces_detected = 0
            
            # معالجة كل وجه تم اكتشافه
            for detection in results.detections:
                # التحقق من نسبة الثقة
                if detection.score[0] < 0.75:
                    continue
                
                # استخراج إحداثيات الوجه
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # التحقق من صحة الإحداثيات
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                
                # حساب مركز ونصف قطر الدائرة
                center_x = x + w // 2
                center_y = y + h // 2
                radius = int(max(w, h) * 0.7)  # تغطية كاملة للوجه
                
                # إنشاء قناع دائري
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
                # تنعيم حواف القناع
                mask = cv2.GaussianBlur(mask, (21, 21), 11)
                
                # تحديد منطقة الوجه
                y1 = max(0, center_y - radius)
                y2 = min(height, center_y + radius)
                x1 = max(0, center_x - radius)
                x2 = min(width, center_x + radius)
                
                if y2 > y1 and x2 > x1:
                    # تمويه منطقة الوجه
                    face_region = result[y1:y2, x1:x2]
                    blurred_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                    
                    # تطبيق القناع
                    mask_region = mask[y1:y2, x1:x2]
                    mask_region = mask_region[:, :, np.newaxis] / 255.0
                    
                    # دمج المنطقة المموهة
                    result[y1:y2, x1:x2] = (
                        face_region * (1 - mask_region) + 
                        blurred_region * mask_region
                    )
                    
                    faces_detected += 1
            
            if faces_detected > 0:
                st.success(TRANSLATIONS['ar']['faces_found'].format(faces_detected))
            
            return Image.fromarray(result)
            
    except Exception as e:
        logger.error(f"خطأ في معالجة الصورة: {str(e)}")
        st.error(TRANSLATIONS['ar']['processing_error'])
        return image

def blur_faces_simple(image):
    """
    تمويه الوجوه مع تحسين الدقة
    """
    try:
        img_array = np.array(image)
        
        # كشف الوجوه
        filtered_faces, _ = detect_faces_advanced(image)
        
        # تمويه كل وجه
        for (x, y, w, h) in filtered_faces:
            # تقليل حجم منطقة التمويه
            padding = int(min(w, h) * 0.05)  # تقليل التمويه الزائد
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_array.shape[1], x + w + padding)
            y2 = min(img_array.shape[0], y + h + padding)
            
            face_roi = img_array[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            img_array[y1:y2, x1:x2] = blurred_face
        
        if not filtered_faces:
            st.warning("⚠️ لم يتم العثور على وجوه في الصورة")
        else:
            st.success(f"✅ تم العثور على {len(filtered_faces)} وجه/وجوه")
            
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

def process_pdf(pdf_file):
    """
    معالجة ملف PDF وتمويه الوجوه في كل صفحة
    """
    try:
        if not PDF_SUPPORT:
            st.error(TRANSLATIONS['ar']['pdf_not_available'])
            return

        with st.spinner(TRANSLATIONS['ar']['pdf_processing']):
            try:
                # تحويل PDF إلى صور
                images = convert_from_bytes(
                    pdf_file.read(),
                    dpi=200,
                    fmt='ppm',
                    thread_count=4
                )
            except Exception as e:
                logger.error(f"خطأ في تحويل PDF: {str(e)}")
                st.error(TRANSLATIONS['ar']['pdf_error'])
                return

            if not images:
                st.warning(TRANSLATIONS['ar']['no_pages'])
                return

            # إنشاء قائمة للصور المعالجة
            processed_images = []
            total_pages = len(images)

            # التحقق من عدد الصفحات
            if total_pages > 100:
                st.warning(TRANSLATIONS['ar']['page_limit'])
                images = images[:100]
                total_pages = 100

            # إنشاء شريط التقدم
            progress_bar = st.progress(0)
            status_text = st.empty()

            # معالجة كل صفحة
            for i, image in enumerate(images):
                try:
                    # تحديث شريط التقدم
                    progress = (i + 1) / total_pages
                    progress_bar.progress(progress)
                    status_text.text(TRANSLATIONS['ar']['processing_page'].format(i+1, total_pages))

                    # تحويل الصورة إلى RGB إذا لزم الأمر
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # معالجة الصورة
                    processed_image = detect_and_blur_faces(image)
                    processed_images.append(processed_image)

                    # تنظيف الذاكرة
                    gc.collect()

                except Exception as e:
                    logger.error(f"خطأ في معالجة الصفحة {i+1}: {str(e)}")
                    processed_images.append(image)
                    continue

            # إزالة شريط التقدم والنص
            progress_bar.empty()
            status_text.empty()

            if not processed_images:
                st.error(TRANSLATIONS['ar']['processing_error'])
                return

            try:
                # إنشاء ملف PDF جديد
                output_pdf = io.BytesIO()
                processed_images[0].save(
                    output_pdf,
                    'PDF',
                    save_all=True,
                    append_images=processed_images[1:],
                    resolution=200.0,
                    quality=95
                )

                st.success(TRANSLATIONS['ar']['success'])

                # زر تحميل الملف المعالج
                st.download_button(
                    TRANSLATIONS['ar']['download_pdf'],
                    output_pdf.getvalue(),
                    "processed_document.pdf",
                    "application/pdf"
                )

            except Exception as e:
                logger.error(f"خطأ في حفظ PDF: {str(e)}")
                st.error(TRANSLATIONS['ar']['save_error'])

    except Exception as e:
        logger.error(f"خطأ في معالجة ملف PDF: {str(e)}")
        st.error(TRANSLATIONS['ar']['processing_error'])

def load_css():
    st.markdown("""
    <style>
    /* تنسيق عام للتطبيق */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        font-family: 'Tajawal', 'Plus Jakarta Sans', sans-serif;
    }
    
    /* إخفاء العناصر غير المرغوبة */
    #MainMenu, header, footer, .stDeployButton {
        display: none !important;
    }
    
    /* العنوان الرئيسي */
    .main-title {
        background: linear-gradient(45deg, #3B82F6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    /* تأثير التوهج */
    @keyframes glow {
        from {
            text-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        }
        to {
            text-shadow: 0 0 30px rgba(59, 130, 246, 0.8);
        }
    }
    
    /* منطقة رفع الملفات */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3B82F6;
        background: rgba(59, 130, 246, 0.1);
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.2);
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
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* تنسيق الصور */
    [data-testid="stImage"] {
        border-radius: 20px;
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
        border-radius: 15px !important;
        backdrop-filter: blur(10px);
        color: white !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* تنسيق النصوص */
    .stMarkdown {
        color: #E2E8F0;
    }
    
    h1, h2, h3 {
        color: #F8FAFC;
        font-weight: 700;
    }
    
    /* تنسيق محدد اللغة */
    .stSelectbox [data-testid="stMarkdown"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* تنسيق شريط التقدم */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3B82F6, #60A5FA);
        height: 8px;
        border-radius: 4px;
    }
    
    /* تأثيرات خلفية */
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
    
    /* تنسيق الأعمدة */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="column"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* تنسيق النص العربي */
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
        line-height: 1.8;
        color: #E2E8F0;
    }
    
    /* تنسيق النص الإنجليزي */
    .english-text {
        direction: ltr;
        text-align: left;
        font-family: 'Plus Jakarta Sans', sans-serif;
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
    
    /* تأثيرات التحميل */
    .stSpinner {
        border-color: #3B82F6 !important;
    }
    
    /* تنسيق التوضيحات */
    .stTooltipIcon {
        color: #3B82F6 !important;
    }
    </style>
    
    <!-- إضافة الخطوط -->
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap" rel="stylesheet">
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
        'pdf_processing': '🔄 جاري معالجة الملف...',
        'pdf_complete': '✅ تمت معالجة جميع الصفحات!',
        'download_pdf': '⬇️ تحميل الملف المعالج (PDF)',
        'notes': '📝 ملاحظات',
        'note_formats': 'يمكنك رفع صور بصيغ JPG, JPEG, PNG أو ملف PDF',
        'note_pdf': 'معالجة ملفات PDF قد تستغرق بعض الوقت حسب عدد الصفحات',
        'processing_error': '❌ حدث خطأ أثناء المعالجة',
        'pdf_not_available': '❌ عذراً، دعم ملفات PDF غير متوفر حالياً',
        'app_error': '❌ حدث خطأ في التطبيق. يرجى المحاولة مرة أخرى',
        'pdf_error': '❌ حدث خطأ في قراءة ملف PDF. تأكد من أن الملف صالح.',
        'no_pages': '⚠️ لم يتم العثور على صفحات في الملف',
        'page_limit': '⚠️ يمكن معالجة 100 صفحة كحد أقصى. سيتم معالجة أول 100 صفحة.',
        'processing_page': 'جاري معالجة الصفحة {} من {}',
        'success': '✅ تمت المعالجة بنجاح',
        'save_error': '❌ حدث خطأ في حفظ الملف النهائي'
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
        'pdf_processing': '🔄 Processing file...',
        'pdf_complete': '✅ All pages processed!',
        'download_pdf': '⬇️ Download Processed File (PDF)',
        'notes': '📝 Notes',
        'note_formats': 'You can upload JPG, JPEG, PNG images or PDF files',
        'note_pdf': 'Processing PDF files may take some time depending on the number of pages',
        'processing_error': '❌ Error during processing',
        'pdf_not_available': '❌ Sorry, PDF support is currently not available',
        'app_error': '❌ Application error occurred. Please try again',
        'pdf_error': '❌ Error reading PDF file. Please make sure the file is valid.',
        'no_pages': '⚠️ No pages found in the file',
        'page_limit': '⚠️ Maximum 100 pages can be processed. Processing first 100 pages.',
        'processing_page': 'Processing page {} of {}',
        'success': '✅ Processing completed successfully',
        'save_error': '❌ Error saving the final file'
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

def process_image(image):
    """
    معالجة الصورة مع محاولات متعددة للكشف
    """
    try:
        # محاولة الكشف باستخدام النموذج المتقدم
        result = detect_and_blur_faces(image)
        
        # إذا لم يتم العثور على وجوه، نجرب النموذج البسيط
        if result == image:
            result = blur_faces_simple(image)
            
        return result
    except Exception as e:
        logger.error(f"خطأ في معالجة الصورة: {str(e)}")
        return image

def main():
    try:
        load_css()
        
        # العنوان الرئيسي
        st.markdown('<div class="main-title">🎭 أداة تمويه الوجوه</div>', unsafe_allow_html=True)
        
        # محدد اللغة
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            lang = st.selectbox(
                "🌐",
                ['ar', 'en'],
                format_func=lambda x: 'العربية' if x == 'ar' else 'English',
                label_visibility="collapsed"
            )
        
        st.markdown("---")
        
        # منطقة رفع الملفات
        uploaded_file = st.file_uploader(
            TRANSLATIONS[lang]['upload_button'],
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help=TRANSLATIONS[lang]['upload_help']
        )
        
        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                if file_extension in ['jpg', 'jpeg', 'png']:
                    # معالجة الصورة
                    image = Image.open(uploaded_file)
                    
                    # عرض الصور في عمودين
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f'<p class="section-title">{TRANSLATIONS[lang]["original_image"]}</p>', unsafe_allow_html=True)
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.markdown(f'<p class="section-title">{TRANSLATIONS[lang]["processed_image"]}</p>', unsafe_allow_html=True)
                        processed_image = detect_and_blur_faces(image)
                        st.image(processed_image, use_container_width=True)
                        
                        # زر التحميل
                        if processed_image is not None:
                            buf = io.BytesIO()
                            processed_image.save(buf, format="PNG")
                            st.download_button(
                                TRANSLATIONS[lang]['download_button'],
                                buf.getvalue(),
                                "blurred_image.png",
                                "image/png"
                            )
                
                elif file_extension == 'pdf' and PDF_SUPPORT:
                    process_pdf(uploaded_file)
                    
                else:
                    st.error(TRANSLATIONS[lang]['format_error'])
                    
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                st.error(TRANSLATIONS[lang]['processing_error'])
        
        # الملاحظات
        st.markdown("---")
        st.markdown(f"""
        <div class="notes-section">
            <h3>{TRANSLATIONS[lang]['notes']}</h3>
            <ul>
                <li>{TRANSLATIONS[lang]['note_formats']}</li>
                {f'<li>{TRANSLATIONS[lang]["note_pdf"]}</li>' if PDF_SUPPORT else ''}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(TRANSLATIONS[lang]['app_error'])

if __name__ == "__main__":
    main()
