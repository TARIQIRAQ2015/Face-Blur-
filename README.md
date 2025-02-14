# أداة تمويه الوجوه

## المتطلبات

1. تثبيت Poppler:
   ```bash
   # على Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y poppler-utils

   # على macOS
   brew install poppler

   # على Windows
   # قم بتحميل وتثبيت Poppler من:
   # http://blog.alivate.com.au/poppler-windows/
   ```

2. تثبيت المكتبات البرمجية:
   ```bash
   pip install -r requirements.txt
   ```

## تشغيل التطبيق
```bash
streamlit run app.py
``` 