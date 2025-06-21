# app.py

from flask import Flask, request, jsonify, abort
# افترض أن دوالك موجودة في model_forecast.py
from model_forecast import load_prediction_assets, make_prediction
import logging

# --- إعداد التطبيق ---
app = Flask(__name__)
# إعداد تسجيل الأخطاء (Logging)
logging.basicConfig(level=logging.INFO)

# --- تحميل أصول النموذج عند بدء التشغيل ---
# هذا الجزء يحتاج إلى تعديل ليتناسب مع كيفية تحميل ملفاتك
# سنستخدم try-except للتعامل مع فشل التحميل عند البدء
try:
    # يجب أن توفر المسارات الصحيحة هنا
    # هذه مجرد أمثلة
    CONFIG_PATH = 'config/config_nn.json'
    FEATURES_PATH = 'config/features.json' # افترضنا وجود هذا الملف
    MODEL_PATH = 'models/lstm_16062025.pth'
    VALID_DATA_PATH = 'valid_16062025.csv'
    MODEL_TYPE = 'lstm'
    TARGET_COIN = 'btc'
    
    # تحميل كل شيء في متغير عام واحد
    assets = load_prediction_assets(CONFIG_PATH, FEATURES_PATH, MODEL_PATH, MODEL_TYPE, VALID_DATA_PATH, TARGET_COIN)
    app.logger.info("--- Model assets loaded successfully! ---")

except Exception as e:
    app.logger.error(f"FATAL: Could not load model assets on startup: {e}")
    assets = None # للدلالة على أن التحميل فشل


# --- نقاط النهاية (Endpoints) ---

@app.route('/health', methods=['GET'])
def health_check():
    """نقطة نهاية للتحقق من صحة الخدمة."""
    # إذا لم يتم تحميل الأصول، تعتبر الخدمة غير صحية
    if assets is None:
        return jsonify({"status": "unhealthy", "message": "Model assets failed to load."}), 503
    return jsonify({"status": "ok", "message": "Service is running and assets are loaded."}), 200

@app.route('/info', methods=['GET'])
def model_info():
    """إرجاع معلومات عن النموذج المستخدم حالياً."""
    if assets and 'model_info' in assets:
        return jsonify(assets['model_info']), 200
    else:
        # Service Unavailable
        return jsonify({"error": "Information not available", "message": "Model assets might not be loaded."}), 503

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """نقطة النهاية الرئيسية لعمل التنبؤ."""
    # التأكد من أن الأصول تم تحميلها
    if assets is None:
        return jsonify({"error": "Service Unavailable", "message": "The prediction service is not ready. Model assets failed to load."}), 503
        
    # التحقق من المدخلات
    if not request.is_json:
        abort(400, description="Invalid request format. Expecting a JSON body.")

    data = request.get_json()
    # يمكنك إضافة المزيد من التحققات هنا، مثلاً على نوع البيانات أو نطاقها
    # ...

    try:
        # افترض أن make_prediction تستقبل القاموس assets والبيانات data
        prediction = make_prediction(assets=assets, input_data=data)
        return jsonify({"prediction": prediction})

    except KeyError as e:
        # في حال كان حقل مطلوب غير موجود في الـ JSON
        abort(400, description=f"Missing required field in request: {e}")
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction: {e}")
        # خطأ خادم داخلي عام
        abort(500)


# --- معالجات الأخطاء (Error Handlers) ---

@app.errorhandler(400)
def bad_request(error):
    """معالج مخصص لخطأ 400 (طلب سيئ)."""
    return jsonify({"error": "Bad Request", "message": error.description or "Invalid data received."}), 400

@app.errorhandler(404)
def not_found(error):
    """معالج مخصص لخطأ 404 (غير موجود)."""
    return jsonify({"error": "Not Found", "message": "This resource does not exist."}), 404

@app.errorhandler(500)
def internal_server_error(error):
    """معالج مخصص لخطأ 500 (خطأ داخلي في الخادم)."""
    return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred on our end."}), 500

@app.errorhandler(503)
def service_unavailable(error):
    """معالج مخصص لخطأ 503 (خدمة غير متوفرة)."""
    return jsonify({"error": "Service Unavailable", "message": error.description or "The service is not ready to handle requests."}), 503


# --- تشغيل التطبيق ---
if __name__ == '__main__':
    # هذا السطر للتشغيل المحلي فقط، Gunicorn هو من سيقوم بالتشغيل داخل Docker
    app.run(debug=True, port=5000)