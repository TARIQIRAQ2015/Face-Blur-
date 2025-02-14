# Face Blur Tool

أداة لتمويه الوجوه في الصور وملفات PDF بشكل تلقائي.

## المتطلبات

1. تثبيت المتطلبات الأساسية:
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr libtesseract-dev libpoppler-cpp-dev python3-dev build-essential
```

2. تثبيت مكتبات Python:
```bash
pip install -r requirements.txt
```

## المميزات

- تمويه الوجوه تلقائياً في الصور
- دعم ملفات PDF
- تمويه دائري للوجوه
- واجهة مستخدم سهلة
- دعم اللغة العربية والإنجليزية

## الاستخدام

1. قم بتشغيل التطبيق:
```bash
streamlit run app.py
```

2. ارفع صورة أو ملف PDF
3. انتظر المعالجة
4. قم بتحميل النتيجة

## الترخيص

MIT License
