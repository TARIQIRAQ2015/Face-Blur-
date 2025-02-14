import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from pdf2image import convert_from_bytes
from PIL import Image
import io
from typing import List, Tuple
import logging

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
            # تحويل الصورة إلى نمط RGB
            img = np.array(image)
            
            # تحويل الصورة إلى BGR لـ OpenCV
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # كشف الوجوه
            results = self.face_detection.process(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            
            if not results.detections:
                logger.info("لم يتم العثور على وجوه في الصورة")
                return image
            
            height, width = img.shape[:2]
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # تحويل الإحداثيات النسبية إلى إحداثيات فعلية
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # التأكد من أن الإحداثيات ضمن حدود الصورة
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                # تطبيق التمويه على منطقة الوجه
                face = img[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(face, self.blur_kernel, self.blur_sigma)
                img[y:y+h, x:x+w] = blurred_face
            
            logger.info(f"تم العثور على {len(results.detections)} وجه/وجوه")
            return Image.fromarray(img)
        
        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {str(e)}")
            raise

def process_pdf(pdf_bytes: io.BytesIO, processor: FaceBlurProcessor) -> List[Image.Image]:
    """
    معالجة ملف PDF وتطبيق التمويه على كل صفحة
    
    Args:
        pdf_bytes: ملف PDF كـ BytesIO
        processor: معالج تمويه الوجوه
    Returns:
        قائمة من الصور المعالجة
    """
    try:
        images = convert_from_bytes(pdf_bytes.read())
        return [processor.blur_faces(img) for img in images]
    except Exception as e:
        logger.error(f"خطأ في معالجة ملف PDF: {str(e)}")
        raise

def main():
    st.set_page_config(
        page_title="أداة تمويه الوجوه",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("🎭 أداة تمويه الوجوه باستخدام الذكاء الاصطناعي")
    
    processor = FaceBlurProcessor()
    
    with st.container():
        uploaded_file = st.file_uploader(
            "ارفع صورة أو ملف PDF",
            type=["jpg", "jpeg", "png", "pdf"],
            help="يمكنك رفع ملفات بصيغة JPG, JPEG, PNG أو PDF"
        )
    
    if uploaded_file is not None:
        try:
            file_type = uploaded_file.type
            
            with st.spinner("جاري معالجة الملف..."):
                if "pdf" in file_type:
                    st.write("📄 معالجة ملف PDF...")
                    processed_images = process_pdf(uploaded_file, processor)
                    for idx, img in enumerate(processed_images, 1):
                        st.image(img, caption=f"صفحة {idx} بعد المعالجة")
                        
                        # زر تحميل لكل صفحة
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        st.download_button(
                            f"تحميل الصفحة {idx}",
                            buf.getvalue(),
                            f"blurred_page_{idx}.png",
                            "image/png"
                        )
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="الصورة الأصلية")
                    
                    with col2:
                        processed_image = processor.blur_faces(image)
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
            st.error(f"حدث خطأ أثناء معالجة الملف: {str(e)}")
            logger.error(f"خطأ: {str(e)}")

if __name__ == "__main__":
    main()
