import streamlit as st
import cv2
import numpy as np
import face_recognition
from pdf2image import convert_from_bytes
from PIL import Image
import io

def blur_faces(image):
    img = np.array(image)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    face_locations = face_recognition.face_locations(rgb_img)
    
    for (top, right, bottom, left) in face_locations:
        face = img[top:bottom, left:right]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        img[top:bottom, left:right] = blurred_face
    
    return Image.fromarray(img)

def process_pdf(pdf_bytes):
    images = convert_from_bytes(pdf_bytes.read())
    processed_images = [blur_faces(img) for img in images]
    return processed_images

def main():
    st.title("أداة تمويه الوجوه باستخدام الذكاء الاصطناعي")
    
    uploaded_file = st.file_uploader("ارفع صورة أو ملف PDF", type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        if "pdf" in file_type:
            st.write("📄 معالجة ملف PDF...")
            processed_images = process_pdf(uploaded_file)
            for img in processed_images:
                st.image(img, caption="صورة معالجة من PDF")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="الصورة الأصلية")
            processed_image = blur_faces(image)
            st.image(processed_image, caption="الصورة بعد التمويه")
            
            buf = io.BytesIO()
            processed_image.save(buf, format="PNG")
            st.download_button("تحميل الصورة المعدلة", buf.getvalue(), "blurred_image.png", "image/png")

if __name__ == "__main__":
    main()
