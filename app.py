import streamlit as st
import cv2
import numpy as np
import face_recognition
from pdf2image import convert_from_bytes
from PIL import Image
import io
from typing import List, Tuple
import time
import sys
import os

# التأكد من تثبيت المتطلبات
def check_dependencies():
    try:
        import face_recognition
        import cv2
        from pdf2image import convert_from_bytes
    except ImportError as e:
        st.error(f"خطأ في تحميل المكتبات المطلوبة: {str(e)}")
        st.info("الرجاء تثبيت المتطلبات باستخدام: pip install -r requirements.txt")
        sys.exit(1)

def configure_page():
    try:
        st.set_page_config(
            page_title="أداة تمويه الوجوه",
            page_icon="👤",
            layout="wide"
        )
        
        st.markdown("""
            <style>
            .main {
                padding: 2rem;
            }
            .stButton>button {
                width: 100%;
            }
            </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"خطأ في تهيئة الصفحة: {str(e)}")

def blur_faces(image: Image.Image, blur_intensity: int = 99) -> Image.Image:
    try:
        # التحقق من صحة الصورة
        if image is None:
            raise ValueError("الصورة غير صالحة")
            
        img = np.array(image)
        if img.size == 0:
            raise ValueError("الصورة فارغة")
            
        # تحويل الصورة إلى BGR
        if len(img.shape) == 2:  # صورة رمادية
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # اكتشاف الوجوه
        face_locations = face_recognition.face_locations(
            rgb_img, 
            model="hog",  # استخدام HOG بشكل افتراضي لأنه أسرع
            number_of_times_to_upsample=1  # زيادة هذا الرقم يحسن الدقة ولكن يبطئ العملية
        )
        
        if not face_locations:
            st.warning("⚠️ لم يتم العثور على وجوه في الصورة")
            return Image.fromarray(img)
        
        # تمويه الوجوه
        for (top, right, bottom, left) in face_locations:
            # التأكد من صحة الإحداثيات
            top = max(0, top)
            right = min(img.shape[1], right)
            bottom = min(img.shape[0], bottom)
            left = max(0, left)
            
            face = img[top:bottom, left:right]
            if face.size > 0:  # التأكد من أن منطقة الوجه صالحة
                # جعل حجم التمويه دائماً فردياً
                blur_size = blur_intensity + (1 - blur_intensity % 2)
                blurred_face = cv2.GaussianBlur(face, (blur_size, blur_size), 30)
                img[top:bottom, left:right] = blurred_face
        
        return Image.fromarray(img)
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
        return image

def process_pdf(pdf_bytes: io.BytesIO, blur_intensity: int) -> List[Image.Image]:
    try:
        # التحقق من صحة الملف
        if pdf_bytes is None or pdf_bytes.getvalue() == b'':
            raise ValueError("ملف PDF غير صالح")
            
        images = convert_from_bytes(pdf_bytes.read())
        if not images:
            raise ValueError("لم يتم العثور على صفحات في ملف PDF")
            
        processed_images = []
        progress_bar = st.progress(0)
        
        for idx, img in enumerate(images):
            processed_images.append(blur_faces(img, blur_intensity))
            progress_bar.progress((idx + 1) / len(images))
        
        progress_bar.empty()
        return processed_images
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة ملف PDF: {str(e)}")
        return []

def main():
    try:
        # التحقق من المتطلبات
        check_dependencies()
        
        # تهيئة الصفحة
        configure_page()
        
        st.title("🎭 أداة تمويه الوجوه باستخدام الذكاء الاصطناعي")
        st.markdown("---")
        
        # إعدادات متقدمة
        with st.expander("⚙️ الإعدادات المتقدمة"):
            col1, col2 = st.columns(2)
            with col1:
                blur_intensity = st.slider("شدة التمويه", 25, 199, 99, step=2)
            with col2:
                st.checkbox("استخدام نموذج CNN (أدق ولكن أبطأ)", key="use_cnn")
        
        uploaded_file = st.file_uploader(
            "📤 ارفع صورة أو ملف PDF",
            type=["jpg", "jpeg", "png", "pdf"],
            help="يمكنك رفع صور بصيغ JPG, JPEG, PNG أو ملفات PDF"
        )
        
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.type
                
                with st.spinner("جاري معالجة الملف..."):
                    if "pdf" in file_type:
                        st.info("🔄 جاري معالجة ملف PDF...")
                        processed_images = process_pdf(uploaded_file, blur_intensity)
                        
                        if processed_images:
                            for idx, img in enumerate(processed_images):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(img, caption=f"صفحة {idx + 1} بعد التمويه")
                                with col2:
                                    buf = io.BytesIO()
                                    img.save(buf, format="PNG")
                                    st.download_button(
                                        f"⬇️ تحميل الصفحة {idx + 1}",
                                        buf.getvalue(),
                                        f"blurred_page_{idx + 1}.png",
                                        "image/png"
                                    )
                    else:
                        image = Image.open(uploaded_file)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="الصورة الأصلية")
                        
                        processed_image = blur_faces(image, blur_intensity)
                        
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
                st.error(f"حدث خطأ غير متوقع: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 📝 ملاحظات:")
        st.markdown("""
            - يمكنك رفع صور بصيغ JPG, JPEG, PNG أو ملفات PDF
            - استخدم الإعدادات المتقدمة للتحكم في جودة التمويه
            - في حالة الملفات الكبيرة، قد تستغرق المعالجة بعض الوقت
        """)
        
    except Exception as e:
        st.error(f"حدث خطأ في تشغيل التطبيق: {str(e)}")

if __name__ == "__main__":
    main()
