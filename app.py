import streamlit as st
import pandas as pd
from PIL import Image
import base64
import os

# في بداية الملف، أضف هذا المتغير
DEFAULT_LOGO = """iVBORw0KGgoAAAANSUhEUgAA... """  # سيتم وضع رمز Base64 للصورة هنا

# تعريف أيقونة افتراضية في حالة عدم وجود الملف
try:
    icon = Image.open('assets/logo.png')
    icon_base64 = base64.b64encode(open('assets/logo.png', 'rb').read()).decode()
except FileNotFoundError:
    # استخدام الصورة الافتراضية
    icon_base64 = DEFAULT_LOGO
    icon = None

st.set_page_config(
    page_title="المساعد لحساب الوزاري",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# إخفاء جميع العناصر الافتراضية
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    .css-eh5xgm {visibility: hidden;}
    .css-1dp5vir {visibility: hidden;}
    .css-1wrcr25 {display: none;}
    .css-6qob1r {display: none;}
    .css-zt5igj {display: none;}
    .stDeployButton {display:none;}
    div[data-testid="stDecoration"] {display:none;}
    div[data-testid="stMarkdownContainer"] > p {margin: 0;}
    
    @keyframes gradientFlow {
        0% {
            background-position: 0% 50%;
            background-color: #00092a;
        }
        10% {
            background-color: #000829;
        }
        20% {
            background-color: #010a2b;
        }
        30% {
            background-color: #000b2b;
        }
        40% {
            background-color: #00082c;
        }
        50% {
            background-position: 100% 50%;
            background-color: #02082a;
        }
        60% {
            background-color: #010a29;
        }
        70% {
            background-color: #000928;
        }
        80% {
            background-color: #01092d;
        }
        90% {
            background-color: #020b2c;
        }
        100% {
            background-position: 0% 50%;
            background-color: #00092a;
        }
    }

    @keyframes shine {
        0% {
            background-position: -100% 50%;
            opacity: 0.3;
        }
        100% {
            background-position: 200% 50%;
            opacity: 0.6;
        }
    }

    @keyframes float {
        0% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
        100% {
            transform: translateY(0px);
        }
    }

    body {
        animation: gradientFlow 15s ease infinite;
        background-size: 200% 200%;
    }
    
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
        background: rgba(0, 9, 42, 0.8);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 255, 157, 0.2);
        position: relative;
        overflow: hidden;
        animation: float 6s ease-in-out infinite;
        border: 1px solid rgba(0, 255, 157, 0.1);
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 20%,
            rgba(0, 255, 157, 0.1) 40%,
            rgba(0, 255, 157, 0.1) 60%,
            transparent 80%
        );
        animation: shine 4s infinite linear;
        pointer-events: none;
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #fff;
        margin: 0;
        padding: 0;
        letter-spacing: 2px;
        position: relative;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5),
                     0 0 20px rgba(0, 255, 157, 0.3),
                     0 0 30px rgba(0, 255, 157, 0.2);
    }
    
    .app-subtitle {
        font-size: 1.5rem;
        color: #00ff9d;
        margin-top: 1rem;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #00092a, #00ff9d);
        color: #fff;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 157, 0.4);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 20%,
            rgba(0, 255, 157, 0.2) 40%,
            rgba(0, 255, 157, 0.2) 60%,
            transparent 80%
        );
        animation: shine 3s infinite linear;
        pointer-events: none;
    }
    
    .stNumberInput>div>div>input {
        background: rgba(0, 9, 42, 0.9);
        border: 1px solid rgba(0, 255, 157, 0.2);
        color: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 1.1rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.1);
    }
    
    .stNumberInput>div>div>input:focus {
        box-shadow: 0 0 0 2px rgba(0, 255, 157, 0.3);
        border-color: rgba(0, 255, 157, 0.5);
    }
    
    .subject-name {
        color: #00ff9d;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        background: rgba(0, 9, 42, 0.9);
        padding: 1rem;
        border-radius: 10px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.1);
        border: 1px solid rgba(0, 255, 157, 0.1);
        animation: float 6s ease-in-out infinite;
    }
    
    .subject-name::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 20%,
            rgba(0, 255, 157, 0.1) 40%,
            rgba(0, 255, 157, 0.1) 60%,
            transparent 80%
        );
        animation: shine 3s infinite linear;
        pointer-events: none;
    }
    
    .grade-label {
        color: #fff;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        text-shadow: 0 0 5px rgba(0, 255, 157, 0.5);
    }
    
    .results-table {
        background: rgba(0, 9, 42, 0.9);
        border-radius: 15px;
        padding: 1rem;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.1);
        border: 1px solid rgba(0, 255, 157, 0.1);
        animation: float 6s ease-in-out infinite;
    }
    
    .conclusion {
        background: rgba(0, 9, 42, 0.9);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        color: #fff;
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.1);
        border: 1px solid rgba(0, 255, 157, 0.1);
        animation: float 6s ease-in-out infinite;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: rgba(0, 9, 42, 0.9);
        border-radius: 12px;
        color: #fff;
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.1);
        border: 1px solid rgba(0, 255, 157, 0.1);
        animation: float 6s ease-in-out infinite;
    }
    
    .social-links {
        margin-bottom: 1rem;
        display: flex;
        justify-content: center;
        gap: 2rem;
    }
    
    .social-links a {
        color: #00ff9d;
        text-decoration: none;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        background: rgba(0, 255, 157, 0.1);
    }
    
    .social-links a:hover {
        background: rgba(0, 255, 157, 0.2);
        transform: translateY(-2px);
    }
    
    [dir="rtl"] .grade-columns {
        flex-direction: row-reverse;
    }
    
    [dir="ltr"] .grade-columns {
        flex-direction: row;
    }

    /* تحسين شكل حقول الإدخال */
    .stNumberInput > div > div > input {
        background: rgba(0, 9, 42, 0.7) !important;
        color: #00ff9d !important;
        border: 2px solid rgba(0, 255, 157, 0.2) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        font-size: 1.2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        font-weight: bold !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #00ff9d !important;
        box-shadow: 0 0 15px rgba(0, 255, 157, 0.3) !important;
        transform: translateY(-2px);
    }

    /* تحسين شكل عناوين المواد */
    .subject-name {
        color: #00ff9d;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
        padding: 1.2rem;
        background: linear-gradient(45deg, rgba(0, 9, 42, 0.9), rgba(0, 20, 80, 0.9));
        border-radius: 15px;
        border: 2px solid rgba(0, 255, 157, 0.2);
        box-shadow: 0 5px 15px rgba(0, 255, 157, 0.1);
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
    }

    /* تحسين شكل عناوين الفصول */
    .grade-label {
        color: #fff;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.3);
    }

    /* تحسين شكل زر التحليل */
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #000d36, #001f5c) !important;
        color: #00ff9d !important;
        border: 2px solid rgba(0, 255, 157, 0.3) !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        font-weight: bold !important;
        font-size: 1.4rem !important;
        transition: all 0.3s ease !important;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        margin: 2rem 0;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 255, 157, 0.2) !important;
        border-color: #00ff9d !important;
    }

    /* تحسين شكل جدول النتائج */
    .results-table {
        background: rgba(0, 9, 42, 0.8);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid rgba(0, 255, 157, 0.2);
        box-shadow: 0 10px 30px rgba(0, 255, 157, 0.1);
    }

    .dataframe {
        font-size: 1.1rem !important;
        text-align: center !important;
    }

    .dataframe th {
        background: rgba(0, 255, 157, 0.1) !important;
        color: #00ff9d !important;
        padding: 15px !important;
        font-weight: bold !important;
    }

    .dataframe td {
        color: white !important;
        padding: 12px !important;
    }

    /* تحسين شكل قسم النتائج */
    .conclusion {
        background: linear-gradient(45deg, rgba(0, 9, 42, 0.9), rgba(0, 20, 80, 0.9));
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        color: white;
        border: 2px solid rgba(0, 255, 157, 0.2);
        box-shadow: 0 10px 30px rgba(0, 255, 157, 0.1);
        font-size: 1.2rem;
        line-height: 1.8;
    }

    /* تحسين شكل التذييل */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(45deg, rgba(0, 9, 42, 0.9), rgba(0, 20, 80, 0.9));
        border-radius: 20px;
        border: 2px solid rgba(0, 255, 157, 0.2);
    }

    .social-links {
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: center;
        gap: 2rem;
    }

    .social-links a {
        color: #00ff9d;
        text-decoration: none;
        font-weight: bold;
        padding: 1rem 2rem;
        border-radius: 12px;
        background: rgba(0, 255, 157, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 255, 157, 0.2);
    }

    .social-links a:hover {
        background: rgba(0, 255, 157, 0.2);
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 255, 157, 0.2);
    }

    .copyright {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
    }

    /* تعديل ترتيب الأعمدة حسب اللغة */
    .dataframe thead tr {{
        display: flex;
        flex-direction: {direction == 'rtl' and 'row' or 'row-reverse'};
    }}
    
    .dataframe tbody tr {{
        display: flex;
        flex-direction: {direction == 'rtl' and 'row' or 'row-reverse'};
    }}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# اختيار اللغة
language = st.selectbox("", ["العربية", "English"], index=0)

# تحديد اتجاه النص بناءً على اللغة
direction = "rtl" if language == "العربية" else "ltr"

# إضافة العنوان حسب اللغة
if language == "العربية":
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">المساعد لحساب الوزاري</h1>
            <p class="app-subtitle">احسب دخولك للوزاري بدقة وسهولة</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="app-header">
            <h1 class="app-title">Ministry Calculator Assistant</h1>
            <p class="app-subtitle">Calculate your ministry entry accurately and easily</p>
        </div>
    """, unsafe_allow_html=True)

# تحديث CSS للعنوان
st.markdown("""
    <style>
    .app-header {
        margin: 2rem auto;
        padding: 2rem;
        max-width: 1200px;
    }
    
    .app-title {
        color: #00ff9d;
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin: 0;
        text-shadow: 0 0 15px rgba(0, 255, 157, 0.5),
                     0 0 30px rgba(0, 255, 157, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .app-subtitle {
        color: #ffffff;
        font-size: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        font-weight: normal;
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 15px rgba(0, 255, 157, 0.5),
                         0 0 30px rgba(0, 255, 157, 0.3);
        }
        to {
            text-shadow: 0 0 20px rgba(0, 255, 157, 0.7),
                         0 0 40px rgba(0, 255, 157, 0.5);
        }
    }
    </style>
""", unsafe_allow_html=True)

# تحديث CSS للواجهة
st.markdown("""
    <style>
    /* تحسين المظهر العام */
    .stApp {
        background: linear-gradient(135deg, #000428 0%, #004e92 100%);
    }
    
    /* النمط العربي */
    .app-header-ar {
        margin: 2rem auto;
        padding: 2rem;
        max-width: 1200px;
    }
    
    .app-title-ar {
        color: #00ff9d;
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin: 0;
        text-shadow: 0 0 15px rgba(0, 255, 157, 0.5),
                     0 0 30px rgba(0, 255, 157, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .app-subtitle-ar {
        color: #ffffff;
        font-size: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        font-weight: normal;
    }

    /* النمط الإنجليزي */
    .app-header-en {
        margin: 2rem auto;
        padding: 3rem 2rem;
        max-width: 900px;
        background: linear-gradient(135deg, rgba(0, 9, 42, 0.95), rgba(0, 20, 80, 0.95));
        border-radius: 25px;
        border: 2px solid rgba(0, 255, 157, 0.2);
        box-shadow: 0 8px 32px rgba(0, 255, 157, 0.15);
    }
    
    .app-title-en {
        color: #00ff9d;
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 15px rgba(0, 255, 157, 0.5),
                     0 0 30px rgba(0, 255, 157, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .app-subtitle-en {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.6rem;
        text-align: center;
        font-weight: 500;
    }

    /* تأثيرات مشتركة */
    @keyframes glow {
        from {
            text-shadow: 0 0 15px rgba(0, 255, 157, 0.5),
                         0 0 30px rgba(0, 255, 157, 0.3);
        }
        to {
            text-shadow: 0 0 20px rgba(0, 255, 157, 0.7),
                         0 0 40px rgba(0, 255, 157, 0.5);
        }
    }

    /* تحسين النتائج */
    .advice-section {
        background: rgba(0, 9, 42, 0.8);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(0, 255, 157, 0.2);
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.1);
    }

    .advice-item {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: rgba(0, 9, 42, 0.7);
    }

    .advice-item.success {
        border-right: 4px solid #00ff9d;
        background: rgba(0, 255, 157, 0.1);
    }

    .advice-item.warning {
        border-right: 4px solid #ffc107;
        background: rgba(255, 193, 7, 0.1);
    }

    .advice-item.danger {
        border-right: 4px solid #ff4848;
        background: rgba(255, 72, 72, 0.1);
    }

    /* تحسين واجهة المستخدم */
    .stSelectbox {
        background: rgba(0, 9, 42, 0.7);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .stSelectbox > div > div {
        background: transparent !important;
        border: 1px solid rgba(0, 255, 157, 0.2) !important;
        color: #fff !important;
    }

    /* تحسين حقول الإدخال */
    .stNumberInput > div > div > input {
        background: rgba(0, 9, 42, 0.7) !important;
        border: 1px solid rgba(0, 255, 157, 0.2) !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 0.8rem !important;
        font-size: 1.1rem !important;
        text-align: center !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #00ff9d !important;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.3) !important;
    }

    /* تحسين الأزرار */
    .stButton > button {
        background: linear-gradient(45deg, #000428, #004e92) !important;
        color: #00ff9d !important;
        border: 1px solid rgba(0, 255, 157, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 2rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 157, 0.2);
        border-color: #00ff9d !important;
    }

    /* تحسين التذييل */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: rgba(0, 9, 42, 0.8);
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 157, 0.2);
        text-align: center;
    }

    .social-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }

    .social-links a {
        color: #00ff9d;
        text-decoration: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        background: rgba(0, 255, 157, 0.1);
        transition: all 0.3s ease;
        font-weight: bold;
    }

    .social-links a:hover {
        background: rgba(0, 255, 157, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 157, 0.2);
    }

    .copyright {
        color: rgba(255, 255, 255, 0.7);
        margin-top: 1rem;
        font-size: 0.9rem;
    }

    /* تحسينات عامة */
    ::-webkit-scrollbar {
        display: none;
    }

    body {
        overflow-x: hidden !important;
    }

    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* تحسين الجداول */
    .dataframe {
        background: rgba(0, 9, 42, 0.8) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 255, 157, 0.2) !important;
        color: #fff !important;
    }

    .dataframe th {
        background: rgba(0, 255, 157, 0.1) !important;
        color: #00ff9d !important;
        font-weight: bold !important;
        padding: 1rem !important;
    }

    .dataframe td {
        padding: 0.8rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# تعريف النصوص حسب اللغة
texts = {
    "العربية": {
        "first_term": "الفصل الأول",
        "mid_term": "نصف السنة",
        "second_term": "الفصل الثاني",
        "analyze": "تحليل النتائج",
        "subjects": {
            "الإسلامية": "الإسلامية",
            "اللغة العربية": "اللغة العربية",
            "اللغة الإنجليزية": "اللغة الإنجليزية",
            "اللغة الفرنسية": "اللغة الفرنسية",
            "الرياضيات": "الرياضيات",
            "الفيزياء": "الفيزياء",
            "الكيمياء": "الكيمياء",
            "الأحياء": "الأحياء"
        }
    },
    "English": {
        "first_term": "First Term",
        "mid_term": "Mid Term",
        "second_term": "Second Term",
        "analyze": "Analyze Results",
        "subjects": {
            "الإسلامية": "Islamic Studies",
            "اللغة العربية": "Arabic",
            "اللغة الإنجليزية": "English",
            "اللغة الفرنسية": "French",
            "الرياضيات": "Mathematics",
            "الفيزياء": "Physics",
            "الكيمياء": "Chemistry",
            "الأحياء": "Biology"
        }
    }
}

current_texts = texts[language]

# تعريف المواد
subjects = {
    "الإسلامية": {"الفصل الأول": 0, "نصف السنة": 0, "الفصل الثاني": 0},
    "اللغة العربية": {"الفصل الأول": 0, "نصف السنة": 0, "الفصل الثاني": 0},
    "اللغة الإنجليزية": {"الفصل الأول": 0, "نصف السنة": 0, "الفصل الثاني": 0},
    "اللغة الفرنسية": {"الفصل الأول": 0, "نصف السنة": 0, "الفصل الثاني": 0},
    "الرياضيات": {"الفصل الأول": 0, "نصف السنة": 0, "الفصل الثاني": 0},
    "الفيزياء": {"الفصل الأول": 0, "نصف السنة": 0, "الفصل الثاني": 0},
    "الكيمياء": {"الفصل الأول": 0, "نصف السنة": 0, "الفصل الثاني": 0},
    "الأحياء": {"الفصل الأول": 0, "نصف السنة": 0, "الفصل الثاني": 0}
}

# إدخال الدرجات
for subject in subjects:
    st.markdown(f'<div class="subject-name">{current_texts["subjects"][subject]}</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    
    with cols[0]:
        st.markdown(f'<div class="grade-label">{current_texts["first_term"]}</div>', unsafe_allow_html=True)
        subjects[subject]["الفصل الأول"] = st.number_input(
            "",
            value=int(subjects[subject]["الفصل الأول"]),
            min_value=0,
            max_value=100,
            key=f"first_{subject}"
        )
    
    with cols[1]:
        st.markdown(f'<div class="grade-label">{current_texts["mid_term"]}</div>', unsafe_allow_html=True)
        subjects[subject]["نصف السنة"] = st.number_input(
            "",
            value=int(subjects[subject]["نصف السنة"]),
            min_value=0,
            max_value=100,
            key=f"mid_{subject}"
        )
    
    with cols[2]:
        st.markdown(f'<div class="grade-label">{current_texts["second_term"]}</div>', unsafe_allow_html=True)
        subjects[subject]["الفصل الثاني"] = st.number_input(
            "",
            value=int(subjects[subject]["الفصل الثاني"]),
            min_value=0,
            max_value=100,
            key=f"second_{subject}"
        )

def calculate_minimum_required(first_term, mid_term):
    required_total = 50 * 3
    current_total = first_term + mid_term
    minimum_required = required_total - current_total
    return minimum_required

if st.button(current_texts["analyze"], key="calculate_btn"):
    results = []
    passing_subjects = []
    impossible_subjects = []  # المواد التي يستحيل النجاح فيها
    need_improvement_subjects = []  # المواد التي تحتاج إلى تحسين
    
    for subject, scores in subjects.items():
        # تضمين اللغة الفرنسية فقط إذا كان هناك درجة واحدة على الأقل
        if subject == "اللغة الفرنسية":
            has_any_grade = scores["الفصل الأول"] > 0 or scores["نصف السنة"] > 0 or scores["الفصل الثاني"] > 0
            if not has_any_grade:
                continue
            
        minimum_required = calculate_minimum_required(
            scores["الفصل الأول"],
            scores["نصف السنة"]
        )
        
        status = ""
        if minimum_required <= 0:
            status = f"✅ ({minimum_required:.0f})"
            passing_subjects.append(subject)
        elif minimum_required > 100:
            status = f"❌ ({minimum_required:.0f})"
            impossible_subjects.append(subject)
        else:
            status = f"❌ ({minimum_required:.0f})"
            need_improvement_subjects.append(subject)
        
        if language == "العربية":
            results.append({
                "المادة": subject,
                "الفصل الأول": scores["الفصل الأول"],
                "نصف السنة": scores["نصف السنة"],
                "الفصل الثاني": scores["الفصل الثاني"],
                "الحد الأدنى المطلوب في الفصل الثاني": status
            })
        else:
            results.append({
                "Subject": current_texts["subjects"][subject],
                "First Term": scores["الفصل الأول"],
                "Mid Term": scores["نصف السنة"],
                "Second Term": scores["الفصل الثاني"],
                "Minimum Required": status
            })
    
    # عند إنشاء DataFrame وعرض النتائج
    if language == "العربية":
        # إعادة ترتيب الأعمدة للغة العربية
        columns = ["المادة", "الفصل الأول", "نصف السنة", "الفصل الثاني", "الحد الأدنى المطلوب في الفصل الثاني"]
    else:
        # ترتيب الأعمدة للغة الإنجليزية
        columns = ["Subject", "First Term", "Mid Term", "Second Term", "Minimum Required"]

    df = pd.DataFrame(results)
    df = df[columns]

    # تحديث CSS للجداول
    st.markdown(f"""
        <style>
        /* تنسيق الجداول */
        .dataframe {{
            direction: {direction};
            background: rgba(0, 9, 42, 0.8) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(0, 255, 157, 0.2) !important;
            color: #fff !important;
            width: 100% !important;
        }}

        /* تنسيق رؤوس الأعمدة */
        .dataframe thead tr th {{
            text-align: {'right' if direction == 'rtl' else 'left'} !important;
            background: rgba(0, 255, 157, 0.1) !important;
            color: #00ff9d !important;
            font-weight: bold !important;
            padding: 1rem !important;
            white-space: nowrap !important;
        }}

        /* تنسيق خلايا الجدول */
        .dataframe tbody tr td {{
            text-align: {'right' if direction == 'rtl' else 'left'} !important;
            padding: 0.8rem !important;
        }}

        /* تنسيق الصفوف */
        .dataframe tbody tr {{
            border-bottom: 1px solid rgba(0, 255, 157, 0.1) !important;
        }}

        /* تنسيق الصف عند التحويم */
        .dataframe tbody tr:hover {{
            background: rgba(0, 255, 157, 0.05) !important;
        }}

        /* تنسيق عمود الترقيم */
        .dataframe .index {{
            text-align: center !important;
            background: rgba(0, 255, 157, 0.05) !important;
            font-weight: bold !important;
        }}

        /* تعديل ترتيب العرض للغة العربية */
        [dir="rtl"] .dataframe {{
            display: table !important;
        }}

        [dir="rtl"] .dataframe thead {{
            float: right !important;
        }}

        [dir="rtl"] .dataframe tbody {{
            float: right !important;
        }}

        [dir="rtl"] .dataframe tr {{
            float: right !important;
        }}

        [dir="rtl"] .dataframe td,
        [dir="rtl"] .dataframe th {{
            float: right !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # عرض الجدول
    st.table(df)
    
    # عرض النصائح في قسم منفصل
    passed_subjects_str = "، ".join(passing_subjects) if passing_subjects else "لا يوجد"
    impossible_subjects_str = "، ".join(impossible_subjects) if impossible_subjects else "لا يوجد"
    need_improvement_subjects_str = "، ".join(need_improvement_subjects) if need_improvement_subjects else "لا يوجد"
    
    # تحديد إمكانية الدخول للوزاري
    total_subjects = len(results)  # عدد المواد الكلي
    passing_count = len(passing_subjects)  # عدد المواد المضمونة
    impossible_count = len(impossible_subjects)  # عدد المواد المستحيلة
    improvement_count = len(need_improvement_subjects)  # عدد المواد التي تحتاج تحسين
    
    # تحديد النصيحة النهائية حسب اللغة
    if language == "العربية":
        if passing_count >= 4:
            final_advice = (
                '<div class="advice-item success final-advice">'
                '🎉 مبارك! يمكنك الدخول للوزاري حيث أنك ضامن النجاح في 4 مواد أو أكثر.'
                '</div>'
            )
        elif passing_count + improvement_count >= 4:
            improvement_details = []
            for subject in need_improvement_subjects:
                min_required = calculate_minimum_required(
                    subjects[subject]["الفصل الأول"],
                    subjects[subject]["نصف السنة"]
                )
                improvement_details.append(f"{subject} (تحتاج {min_required:.0f} درجة)")

            improvement_subjects_details = "، ".join(improvement_details)
            
            final_advice = (
                '<div class="advice-item warning final-advice">'
                f'⚠️ يمكنك الدخول للوزاري مع التركيز على تحسين درجاتك.'
                f'<br>لديك {passing_count} مواد مضمونة.'
                f'<br>المواد التي تحتاج إلى تحسين هي: {improvement_subjects_details}.'
                f'<br>تحتاج إلى النجاح في {max(4 - passing_count, 0)} مواد على الأقل من المواد المتبقية.'
                '</div>'
            )
        else:
            final_advice = (
                '<div class="advice-item danger final-advice">'
                f'⛔ غير مؤهل للدخول للوزاري هذا العام.'
                f'<br>لديك فقط {passing_count} مواد مضمونة و {improvement_count} مواد تحتاج إلى تحسين.'
                f'<br>يجب ضمان النجاح في 4 مواد على الأقل للتأهل للوزاري.'
                '</div>'
            )
    else:
        if passing_count >= 4:
            final_advice = (
                '<div class="advice-item success final-advice">'
                '🎉 Congratulations! You can enter the ministry exam as you have guaranteed success in 4 or more subjects.'
                '</div>'
            )
        elif passing_count + improvement_count >= 4:
            improvement_details = []
            for subject in need_improvement_subjects:
                min_required = calculate_minimum_required(
                    subjects[subject]["الفصل الأول"],
                    subjects[subject]["نصف السنة"]
                )
                improvement_details.append(f"{current_texts['subjects'][subject]} (needs {min_required:.0f} points)")

            improvement_subjects_details = ", ".join(improvement_details)
            
            final_advice = (
                '<div class="advice-item warning final-advice">'
                f'⚠️ You can enter the ministry exam with focus on improving your grades.'
                f'<br>You have {passing_count} guaranteed subjects.'
                f'<br>Subjects that need improvement: {improvement_subjects_details}.'
                f'<br>You need to pass at least {max(4 - passing_count, 0)} subjects from the remaining ones.'
                '</div>'
            )
        else:
            final_advice = (
                '<div class="advice-item danger final-advice">'
                f'⛔ Not eligible for ministry exam this year.'
                f'<br>You only have {passing_count} guaranteed subjects and {improvement_count} subjects need improvement.'
                f'<br>You must guarantee success in at least 4 subjects to qualify.'
                '</div>'
            )

    # تحديث عرض النصائح مع إضافة التقييم النهائي
    if language == "العربية":
        st.markdown(f"""
            <div class="advice-section">
                <div class="advice-item success">
                    ✅ المواد التي ضمنت النجاح هي: {passed_subjects_str} حتى لو حصلت على 0 في الفصل الثاني.
                </div>
                <br>
                <div class="advice-item warning">
                    ⚠️ المواد التي تحتاج إلى تحسين هي: {need_improvement_subjects_str}
                </div>
                <br>
                <div class="advice-item danger">
                    ❌ المواد التي يستحيل النجاح فيها هي: {impossible_subjects_str}
                </div>
                <br>
                <div class="final-advice-separator"></div>
                {final_advice}
            </div>
        """, unsafe_allow_html=True)
    else:
        # تحويل أسماء المواد للإنجليزية
        passed_subjects_en = ", ".join([current_texts["subjects"][sub] for sub in passing_subjects]) if passing_subjects else "None"
        impossible_subjects_en = ", ".join([current_texts["subjects"][sub] for sub in impossible_subjects]) if impossible_subjects else "None"
        need_improvement_subjects_en = ", ".join([current_texts["subjects"][sub] for sub in need_improvement_subjects]) if need_improvement_subjects else "None"
        
        st.markdown(f"""
            <div class="advice-section">
                <div class="advice-item success">
                    ✅ Subjects with guaranteed success: {passed_subjects_en} even if you get 0 in the second term.
                </div>
                <br>
                <div class="advice-item warning">
                    ⚠️ Subjects that need improvement: {need_improvement_subjects_en}
                </div>
                <br>
                <div class="advice-item danger">
                    ❌ Subjects impossible to pass: {impossible_subjects_en}
                </div>
                <br>
                <div class="final-advice-separator"></div>
                {final_advice}
            </div>
        """, unsafe_allow_html=True)

# إضافة CSS للتقييم النهائي
st.markdown("""
    <style>
    .final-advice-separator {
        border-top: 1px solid rgba(0, 255, 157, 0.2);
        margin: 1rem 0;
    }
    
    .final-advice {
        font-size: 1.2rem !important;
        padding: 1.2rem !important;
        margin-top: 1rem !important;
        border-width: 2px !important;
    }
    </style>
""", unsafe_allow_html=True)

# إضافة معلومات التواصل وحقوق النشر
st.markdown("""
    <div class="footer">
        <div class="social-links">
            <a href="https://t.me/SadsHelp" target="_blank">شبكة المساعد التعليمية 📖</a>
            <a href="https://t.me/+mg19Snwv14U4NWZi" target="_blank">كروب طلاب السادس الاعدادي 📖</a>
        </div>
        <div class="copyright">
            By Tariq Al-Yaseen © 2025-2026
        </div>
    </div>
""", unsafe_allow_html=True)

print("Current working directory:", os.getcwd())
print("Logo file exists:", os.path.exists('logo.png'))

# تحديث CSS للنتائج والنصائح
st.markdown(f"""
    <style>
    /* تنسيق قسم النصائح */
    .advice-section {{
        background: rgba(0, 9, 42, 0.8);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(0, 255, 157, 0.2);
        box-shadow: 0 4px 15px rgba(0, 255, 157, 0.1);
        direction: {direction};
        text-align: {'right' if direction == 'rtl' else 'left'};
    }}

    .advice-item {{
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: rgba(0, 9, 42, 0.7);
        border-right: {'4px solid transparent' if direction == 'rtl' else 'none'};
        border-left: {'4px solid transparent' if direction == 'ltr' else 'none'};
    }}

    .advice-item.success {{
        border-right-color: {'#00ff9d' if direction == 'rtl' else 'transparent'};
        border-left-color: {'#00ff9d' if direction == 'ltr' else 'transparent'};
        background: rgba(0, 255, 157, 0.1);
    }}

    .advice-item.warning {{
        border-right-color: {'#ffc107' if direction == 'rtl' else 'transparent'};
        border-left-color: {'#ffc107' if direction == 'ltr' else 'transparent'};
        background: rgba(255, 193, 7, 0.1);
    }}

    .advice-item.danger {{
        border-right-color: {'#ff4848' if direction == 'rtl' else 'transparent'};
        border-left-color: {'#ff4848' if direction == 'ltr' else 'transparent'};
        background: rgba(255, 72, 72, 0.1);
    }}

    /* تنسيق الجداول */
    .dataframe {{
        direction: {direction};
        text-align: {'right' if direction == 'rtl' else 'left'};
        background: rgba(0, 9, 42, 0.8) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 255, 157, 0.2) !important;
        color: #fff !important;
    }}

    .dataframe th {{
        text-align: {'right' if direction == 'rtl' else 'left'} !important;
        background: rgba(0, 255, 157, 0.1) !important;
        color: #00ff9d !important;
        font-weight: bold !important;
        padding: 1rem !important;
    }}

    .dataframe td {{
        text-align: {'right' if direction == 'rtl' else 'left'} !important;
        padding: 0.8rem !important;
    }}

    /* تنسيق التقييم النهائي */
    .final-advice {{
        direction: {direction};
        text-align: {'right' if direction == 'rtl' else 'left'};
        font-size: 1.2rem !important;
        padding: 1.5rem !important;
        margin-top: 1.5rem !important;
        border-width: 2px !important;
        background: rgba(0, 9, 42, 0.9) !important;
    }}

    .final-advice-separator {{
        border-top: 2px solid rgba(0, 255, 157, 0.2);
        margin: 1.5rem 0;
    }}

    /* تنسيق خاص للجدول باللغة العربية */
    [dir="rtl"] .dataframe {{
        direction: rtl !important;
    }}
    
    [dir="rtl"] .dataframe thead tr,
    [dir="rtl"] .dataframe tbody tr {{
        display: flex !important;
        flex-direction: row-reverse !important;
    }}
    
    [dir="rtl"] .dataframe th,
    [dir="rtl"] .dataframe td {{
        flex: 1 !important;
        text-align: right !important;
    }}
    </style>
""", unsafe_allow_html=True)
