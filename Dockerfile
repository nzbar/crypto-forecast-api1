# Dockerfile لتطبيق Flask - نسخة مرنة تستخدم متغير البيئة PORT

# الخطوة 1: ابدأ من صورة بايثون رسمية وخفيفة
FROM python:3.9-slim

# الخطوة 2: تجهيز بيئة العمل بالشكل الموصى به
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# الخطوة 3: تثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# الخطوة 4: انسخ كود المشروع
COPY . .

# الخطوة 5: نعرض المنفذ الافتراضي الذي سيعمل عليه التطبيق
# هذا السطر للمعلومات فقط، ولا يؤثر على التشغيل
EXPOSE 8000

# الخطوة 6: الأمر النهائي والمعدل لتشغيل الخادم
# سيستخدم متغير البيئة PORT إذا كان موجوداً، وإذا لم يكن، سيستخدم 8000 كقيمة افتراضية
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} app:app