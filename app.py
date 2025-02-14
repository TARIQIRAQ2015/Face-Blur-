import streamlit as st
import cv2
import numpy as np
import face_recognition
from pdf2image import convert_from_bytes
from PIL import Image
import io
from typing import List, Tuple
import time

def configure_page():
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

def blur_faces(image: Image.Image, blur_intensity: int = 99) -> Image.Image:
    try:
        img = np.array(image)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # تحسين دقة اكتشاف الوجوه
        face_locations = face_recognition.face_locations(rgb_img, model="cnn" if st.session_state.get('use_cnn', False) else "hog")
        
        if not face_locations:
            st.warning("⚠️ لم يتم العثور على وجوه في الصورة")
            return Image.fromarray(img)
        
        for (top, right, bottom, left) in face_locations:
            face = img[top:bottom, left:right]
            # تحسين جودة التمويه
            blurred_face = cv2.GaussianBlur(face, (blur_intensity, blur_intensity), 30)
            img[top:bottom, left:right] = blurred_face
        
        return Image.fromarray(img)
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
        return image

def process_pdf(pdf_bytes: io.BytesIO, blur_intensity: int) -> List[Image.Image]:
    try:
        images = convert_from_bytes(pdf_bytes.read())
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
    
    uploaded_file = st.file_uploader("📤 ارفع صورة أو ملف PDF", type=["jpg", "jpeg", "png", "pdf"])
    
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

if __name__ == "__main__":
    main()
