import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def main():
    try:
        configure_page()
        
        st.title("🎭 أداة تمويه الوجوه")
        st.markdown("---")
        
        # إعدادات التمويه
        blur_intensity = st.slider("شدة التمويه", 25, 199, 99, step=2)
        
        # رفع الملف
        uploaded_file = st.file_uploader(
            "📤 ارفع صورة",
            type=["jpg", "jpeg", "png"],
            help="يمكنك رفع صور بصيغ JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            try:
                # عرض الصورة الأصلية
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="الصورة الأصلية")
                
                # معالجة الصورة
                with st.spinner("جاري معالجة الصورة..."):
                    processed_image = blur_faces_simple(image)
                
                # عرض الصورة المعالجة
                with col2:
                    st.image(processed_image, caption="الصورة بعد التمويه")
                
                # زر التحميل
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
        - يمكنك رفع صور بصيغ JPG, JPEG, PNG
        - استخدم شريط التمرير للتحكم في شدة التمويه
        """)
        
    except Exception as e:
        logger.error(f"خطأ في التطبيق: {str(e)}")
        st.error(f"حدث خطأ في التطبيق: {str(e)}")

if __name__ == "__main__":
    main()
