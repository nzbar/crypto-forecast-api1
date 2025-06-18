# ----------------------------------------------------
# Dockerfile لتطبيق Flask مع خادم Gunicorn
# (هذا هو الملف المناسب لمشروعك)
# ----------------------------------------------------

# الخطوة 1: ابدأ من صورة بايثون رسمية وخفيفة
FROM python:3.9-slim

# الخطوة 2: تجهيز بيئة العمل
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app

# الخطوة 3: تثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# الخطوة 4: انسخ كود المشروع
COPY . .

# الخطوة 5: تشغيل الخادم
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]