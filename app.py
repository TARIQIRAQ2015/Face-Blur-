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

def detect_faces_advanced(image):
    """
    كشف الوجوه باستخدام خوارزميات متعددة
    """
    try:
        # تحويل الصورة إلى مصفوفة numpy
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # تحميل جميع الكواشف المتوفرة
        cascades = {
            'frontal_default': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'frontal_alt': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
            'frontal_alt2': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
            'profile': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml'),
            'eye': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml'),
            'eye_tree_eyeglasses': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'),
        }
        
        # معلمات الكشف المختلفة للتجربة
        scale_factors = [1.05, 1.1, 1.15]
        min_neighbors_options = [3, 4, 5]
        
        all_faces = []
        
        # تجربة كل كاشف مع معلمات مختلفة
        for cascade_name, cascade in cascades.items():
            if cascade_name.startswith('frontal'):
                for scale_factor in scale_factors:
                    for min_neighbors in min_neighbors_options:
                        faces = cascade.detectMultiScale(
                            gray,
                            scaleFactor=scale_factor,
                            minNeighbors=min_neighbors,
                            minSize=(20, 20),
                            maxSize=(1000, 1000),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        all_faces.extend(faces)
            
            elif cascade_name == 'profile':
                # كشف الوجوه الجانبية في الاتجاهين
                for angle in [0, 1]:  # 0 للصورة الأصلية، 1 للصورة المعكوسة
                    temp_gray = cv2.flip(gray, angle) if angle == 1 else gray
                    for scale_factor in scale_factors:
                        faces = cascade.detectMultiScale(
                            temp_gray,
                            scaleFactor=scale_factor,
                            minNeighbors=4,
                            minSize=(20, 20),
                            maxSize=(1000, 1000)
                        )
                        if angle == 1:
                            faces = [(temp_gray.shape[1] - x - w, y, w, h) for (x, y, w, h) in faces]
                        all_faces.extend(faces)
        
        return all_faces, cascades['eye']
    
    except Exception as e:
        logger.error(f"خطأ في كشف الوجوه: {str(e)}")
        return [], None

def blur_faces_simple(image, blur_intensity=99):
    """
    تمويه الوجوه بشكل دقيق يتناسب مع حدود الوجه
    """
    try:
        img_array = np.array(image)
        
        # كشف الوجوه باستخدام الدالة المتقدمة
        all_faces, eye_cascade = detect_faces_advanced(image)
        
        # إزالة التداخلات
        filtered_faces = remove_overlapping_faces(all_faces)
        
        # تمويه كل وجه
        for (x, y, w, h) in filtered_faces:
            # توسيع منطقة الوجه قليلاً
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_array.shape[1], x + w + padding)
            y2 = min(img_array.shape[0], y + h + padding)
            
            # تمويه منطقة الوجه
            face_roi = img_array[y1:y2, x1:x2]
            kernel_size = int(blur_intensity // 2) * 2 + 1  # ضمان أن يكون الرقم فردياً
            blurred_face = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 30)
            img_array[y1:y2, x1:x2] = blurred_face
            
            # محاولة كشف وتمويه العيون بشكل إضافي
            if eye_cascade is not None:
                eyes = eye_cascade.detectMultiScale(
                    cv2.cvtColor(img_array[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY),
                    scaleFactor=1.1,
                    minNeighbors=4
                )
                for (ex, ey, ew, eh) in eyes:
                    eye_roi = img_array[y+ey:y+ey+eh, x+ex:x+ex+ew]
                    blurred_eye = cv2.GaussianBlur(eye_roi, (51, 51), 15)
                    img_array[y+ey:y+ey+eh, x+ex:x+ex+ew] = blurred_eye
        
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

def load_css():
    st.markdown("""
    <style>
    /* التنسيق العام */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        color: #ffffff;
    }
    
    /* تنسيق الشريط الجانبي */
    [data-testid="stSidebar"] {
        background: rgba(30, 30, 46, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* تنسيق منطقة رفع الملفات */
    .upload-area {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .upload-area:hover {
        border-color: #2196F3;
        background: rgba(33, 150, 243, 0.1);
    }
    
    /* تنسيق شريط التمرير */
    [data-testid="stSlider"] > div > div {
        background: linear-gradient(90deg, #2196F3, #00BCD4);
    }
    
    /* تنسيق الأزرار */
    .stButton button {
        background: linear-gradient(45deg, #2196F3, #00BCD4);
        border-radius: 50px;
        padding: 0.5rem 2rem;
        border: none;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
    }
    
    /* تنسيق التنبيهات */
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: none;
        border-radius: 10px;
        color: white;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* تنسيق النصوص */
    .text-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* تنسيق العناوين */
    h1 {
        background: linear-gradient(45deg, #2196F3, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    h2, h3, h4 {
        color: #ffffff;
        margin: 1rem 0;
    }
    
    /* تنسيق القوائم */
    ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    ul li {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    ul li:before {
        content: "•";
        color: #2196F3;
        font-size: 1.5rem;
        position: absolute;
        left: 0;
        top: -0.2rem;
    }
    
    /* تنسيق الصور */
    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
    }
    
    /* تنسيق شريط التقدم */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2196F3, #00BCD4);
        height: 10px;
        border-radius: 5px;
    }
    
    /* تنسيق النص باللغة العربية */
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif;
        line-height: 1.6;
    }
    
    /* تنسيق النص باللغة الإنجليزية */
    .english-text {
        direction: ltr;
        text-align: left;
        font-family: 'Roboto', sans-serif;
        line-height: 1.6;
    }
    
    /* تنسيق محدد اللغة */
    .language-selector {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* تأثيرات التحميل */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* تنسيق الروابط */
    a {
        color: #2196F3;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    a:hover {
        color: #00BCD4;
        text-decoration: underline;
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
        'blur_intensity': 'شدة التمويه',
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
        'pdf_not_supported': '⚠️ دعم ملفات PDF غير متوفر',
        'pdf_not_available': 'عذراً، دعم ملفات PDF غير متوفر حالياً',
        'note_formats': 'يمكنك رفع صور بصيغ JPG, JPEG, PNG أو ملف PDF',
        'note_slider': 'استخدم شريط التمرير للتحكم في شدة التمويه',
        'note_pdf': 'معالجة ملفات PDF قد تستغرق بعض الوقت حسب عدد الصفحات',
        'about_title': 'معلومات عن التطبيق',
        'about_text': 'هذا التطبيق يوفر خدمة تمويه الوجوه بطريقة دقيقة وفعالة. يمكنك استخدامه لتحسين خصائص الصور، سواء كانت صورة أو ملف PDF. يتميز التطبيق بقدرته على التعامل مع جميع أنواع الصور والملفات المختلفة، مما يجعله مفيدًا بشكل خاص لمستخدمين الصور والمصورين.',
        'app_error': 'حدث خطأ في التطبيق، يرجى المحاولة لاحقًا',
    },
    'en': {
        'title': '🎭 Face Blur Tool',
        'upload_button': '📤 Upload Image or PDF',
        'upload_help': 'You can upload JPG, JPEG, PNG images or PDF files',
        'blur_intensity': 'Blur Intensity',
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
        'pdf_not_supported': '⚠️ PDF support not available',
        'pdf_not_available': 'Sorry, PDF support is not available currently',
        'note_formats': 'You can upload JPG, JPEG, PNG images or PDF files',
        'note_slider': 'Use the slider to control the blur intensity',
        'note_pdf': 'Processing PDF files may take some time depending on the number of pages',
        'about_title': 'About the App',
        'about_text': 'This app provides a precise and efficient face blurring service. You can use it to enhance your images, whether they are photos or PDF files. The app is versatile and can handle all types of images and files, making it especially useful for image and photo enthusiasts.',
        'app_error': 'Application error, please try again later',
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
        
        # اختيار اللغة في الشريط الجانبي مع تنسيق خاص
        st.sidebar.markdown("""
        <div class="language-selector">
            <h3 style="text-align: center; margin-bottom: 1rem;">🌐</h3>
        </div>
        """, unsafe_allow_html=True)
        
        lang = st.sidebar.selectbox(
            "Language / اللغة",
            ['ar', 'en'],
            format_func=lambda x: 'العربية 🇸🇦' if x == 'ar' else 'English 🇺🇸'
        )
        
        # تطبيق اتجاه النص حسب اللغة
        text_class = 'arabic-text' if lang == 'ar' else 'english-text'
        
        # العنوان الرئيسي
        st.markdown(f'<h1 class="{text_class}">{get_text("title", lang)}</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # رسالة دعم PDF
        if not PDF_SUPPORT:
            st.warning(get_text('pdf_not_supported', lang))
        
        # إعدادات التمويه
        with st.container():
            st.markdown(f'<div class="{text_class}">', unsafe_allow_html=True)
            blur_intensity = st.slider(
                get_text('blur_intensity', lang),
                25, 199, 99, step=2
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # منطقة رفع الملفات
        st.markdown(f"""
        <div class="{text_class} upload-area">
            <p style="text-align: center; margin-bottom: 1rem;">
                {get_text('upload_help', lang)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
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
                        st.markdown(f'<p class="{text_class}">{get_text("original_image", lang)}</p>', unsafe_allow_html=True)
                        st.image(image, use_container_width=True)
                    
                    with st.spinner(get_text('processing', lang)):
                        processed_image = blur_faces_simple(image, blur_intensity)
                    
                    with col2:
                        st.markdown(f'<p class="{text_class}">{get_text("processed_image", lang)}</p>', unsafe_allow_html=True)
                        st.image(processed_image, use_container_width=True)
                    
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
        
        # الملاحظات
        st.markdown("---")
        st.markdown(f"""
        <div class="{text_class}">
            <h3>{get_text('notes', lang)}</h3>
            <ul>
                <li>{get_text('note_formats', lang)}</li>
                <li>{get_text('note_slider', lang)}</li>
                {f'<li>{get_text("note_pdf", lang)}</li>' if PDF_SUPPORT else ''}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # معلومات إضافية في الشريط الجانبي
        with st.sidebar:
            st.markdown(f"""
            <div class="{text_class}" style="margin-top: 2rem;">
                <h4>{get_text('about_title', lang)}</h4>
                <p>{get_text('about_text', lang)}</p>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(get_text('app_error', lang))

if __name__ == "__main__":
    main()
