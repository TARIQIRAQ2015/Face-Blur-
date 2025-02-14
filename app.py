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
        """تهيئة نماذج كشف الوجوه المتقدمة"""
        # MediaPipe Face Mesh للكشف الدقيق
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=20,
            refine_landmarks=True,
            min_detection_confidence=0.4
        )
        
        # MediaPipe Face Detection للوجوه البعيدة
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.4
        )
        
        # OpenCV's Deep Neural Network face detector
        self.dnn_face_detector = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt',
            'res10_300x300_ssd_iter_140000.caffemodel'
        )
        
        # Haar Cascade للوجوه الصغيرة
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces_dnn(self, image: np.ndarray) -> List[dict]:
        """كشف الوجوه باستخدام شبكة DNN"""
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), 
            [104, 117, 123], False, False
        )
        
        self.dnn_face_detector.setInput(blob)
        detections = self.dnn_face_detector.forward()
        faces = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype(int)
                faces.append({
                    'box': [x1, y1, x2-x1, y2-y1],
                    'confidence': float(confidence),
                    'source': 'dnn'
                })
        
        return faces
    
    def detect_faces_mediapipe(self, image: np.ndarray) -> List[dict]:
        """كشف الوجوه باستخدام MediaPipe"""
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        
        if results.detections:
            height, width = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                faces.append({
                    'box': [x, y, w, h],
                    'confidence': detection.score[0],
                    'source': 'mediapipe'
                })
        
        return faces
    
    def detect_faces_cascade(self, image: np.ndarray) -> List[dict]:
        """كشف الوجوه باستخدام Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        return [
            {
                'box': list(map(int, box)),
                'confidence': 0.9,
                'source': 'cascade'
            }
            for box in detections
        ]
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """كشف الوجوه باستخدام جميع الطرق المتاحة"""
        all_faces = []
        
        # جمع النتائج من جميع الطرق
        all_faces.extend(self.detect_faces_mediapipe(image))
        all_faces.extend(self.detect_faces_cascade(image))
        try:
            all_faces.extend(self.detect_faces_dnn(image))
        except Exception as e:
            logger.warning(f"خطأ في DNN detector: {str(e)}")
        
        # دمج وتنقية النتائج
        return self.merge_detections(all_faces)
    
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
                iou = self.calculate_iou(box1, box2)
                if iou > 0.5:
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
    
    def apply_advanced_blur(self, image: np.ndarray, box: List[int], confidence: float) -> np.ndarray:
        """تطبيق تمويه متقدم مع تأثيرات متدرجة"""
        x, y, w, h = box
        center = (x + w//2, y + h//2)
        radius = int(max(w, h) * 0.6)
        
        # إنشاء قناع متدرج للتمويه
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        for r in range(radius):
            alpha = 1 - (r / radius)**2  # تدرج غير خطي
            cv2.circle(mask, center, radius - r, alpha, 1)
        
        # تطبيق عدة مستويات من التمويه
        blur_levels = [
            cv2.GaussianBlur(image, (k, k), 0)
            for k in [21, 41, 81]
        ]
        
        result = image.copy()
        for i, blurred in enumerate(blur_levels):
            weight = mask * (1 - i/len(blur_levels))
            weight = np.expand_dims(weight, -1)
            result = result * (1 - weight) + blurred * weight
        
        return result.astype(np.uint8)
    
    def process_image(self, image: Image.Image) -> Image.Image:
        """معالجة الصورة وتمويه الوجوه"""
        try:
            img = np.array(image)
            
            # تحسين جودة الصورة
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # كشف الوجوه
            faces = self.detector.detect_faces(img)
            
            if not faces:
                logger.info("لم يتم العثور على وجوه في الصورة")
                return image
            
            # تطبيق التمويه على كل وجه
            for face in faces:
                img = self.apply_advanced_blur(
                    img,
                    face['box'],
                    face['confidence']
                )
            
            logger.info(f"تم معالجة {len(faces)} وجه/وجوه")
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
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_UJNc2t.json"
    lottie_json = load_lottie_url(lottie_url)
    
    # تصميم الصفحة الرئيسية
    st.markdown("""
    <div class="title-container">
        <h1>🎭 تمويه الوجوه الذكي</h1>
        <p class="info-text">نظام متقدم للكشف عن الوجوه وتمويهها باستخدام الذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض الرسوم المتحركة
    if lottie_json:
        st_lottie(lottie_json, height=200, key="face_animation")
    
    processor = AdvancedFaceBlurProcessor()
    
    # منطقة رفع الملفات
    st.markdown("""
    <div class="upload-container">
        <h3>📁 رفع الملفات</h3>
        <p class="info-text">يمكنك رفع صور بصيغة JPG, JPEG, PNG أو ملفات PDF</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "اختر ملفاً",
        type=["jpg", "jpeg", "png", "pdf"],
        help="اسحب الملف هنا أو انقر لاختيار ملف"
    )
    
    if uploaded_file:
        try:
            file_type = uploaded_file.type
            
            with st.spinner("🔄 جاري معالجة الملف..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if "pdf" in file_type:
                    status_text.text("📄 جاري معالجة ملف PDF...")
                    images = convert_from_bytes(uploaded_file.read())
                    processed_images = []
                    
                    for idx, img in enumerate(images):
                        progress = (idx + 1) / len(images)
                        progress_bar.progress(progress)
                        status_text.text(f"معالجة الصفحة {idx + 1} من {len(images)}")
                        
                        processed = processor.process_image(img)
                        processed_images.append(processed)
                    
                    st.markdown("""
                    <div class="results-container">
                        <h3>📄 النتائج</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for idx, img in enumerate(processed_images, 1):
                        st.markdown(f"#### صفحة {idx}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img, caption=f"الصفحة {idx} بعد المعالجة", use_column_width=True)
                        with col2:
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            st.download_button(
                                f"⬇️ تحميل الصفحة {idx}",
                                buf.getvalue(),
                                f"blurred_page_{idx}.png",
                                "image/png"
                            )
                else:
                    image = Image.open(uploaded_file)
                    
                    st.markdown("""
                    <div class="results-container">
                        <h3>📊 معلومات الصورة</h3>
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
                            "image/png"
                        )
                
                progress_bar.progress(100)
                status_text.empty()
                st.balloons()
                st.success("✨ تمت المعالجة بنجاح!")
        
        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء المعالجة: {str(e)}")
            logger.error(f"خطأ: {str(e)}")

if __name__ == "__main__":
    main()
