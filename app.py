# app.py (النسخة النهائية مع تحميل ديناميكي للنموذج)

from flask import Flask, request, jsonify, abort
from model_forecast import load_prediction_assets, make_prediction
import logging
import os # استيراد مكتبة os للتعامل مع المسارات

# --- إعداد التطبيق ---
app = Flask(__name__)
# إعداد تسجيل الأخطاء (Logging)
logging.basicConfig(level=logging.INFO)


# --- تحميل أصول النموذج عند بدء التشغيل (النسخة المحدثة) ---
assets = None # تعريف المتغير بشكل عام
try:
    app.logger.info("--- [API] بدء عملية تحميل الأصول ---")
    
    # --- === الجزء المعدل لتحميل أحدث نموذج === ---

    # 1. تحديد المسارات الأساسية
    # هذا هو المسار إلى القرص الصلب الدائم في Render
    MODELS_DIR_PERSISTENT = '/data/models' 
    # هذا هو المسار المحلي داخل المشروع للنموذج الافتراضي
    MODELS_DIR_LOCAL = 'models' 
    # الملف الذي يحتوي على اسم أحدث نموذج
    LATEST_MODEL_INFO_FILE = os.path.join(MODELS_DIR_PERSISTENT, 'latest.txt')
    
    model_filename = None
    model_path = None

    # 2. البحث عن مؤشر أحدث نموذج
    if os.path.exists(LATEST_MODEL_INFO_FILE):
        with open(LATEST_MODEL_INFO_FILE, 'r') as f:
            model_filename = f.read().strip()
        app.logger.info(f"تم العثور على مؤشر لأحدث نموذج: '{model_filename}'")
        # بناء المسار الكامل للنموذج من القرص الدائم
        model_path = os.path.join(MODELS_DIR_PERSISTENT, model_filename)
    
    # 3. في حال عدم وجود نموذج جديد، استخدم النموذج الافتراضي
    if not model_path or not os.path.exists(model_path):
        app.logger.warning(f"لم يتم العثور على نموذج جديد في القرص الدائم، سيتم استخدام النموذج الافتراضي.")
        model_filename = 'lstm_16062025.pth' # اسم نموذجك الافتراضي
        model_path = os.path.join(MODELS_DIR_LOCAL, model_filename)

    app.logger.info(f"سيتم تحميل النموذج من المسار: '{model_path}'")
    
    # --- === نهاية الجزء المعدل === ---

    # الآن نستدعي دالة التحميل بالمسار الصحيح والديناميكي
    assets = load_prediction_assets(
        config_path='config/config_nn.json',
        features_path='config/features.json',
        model_path=model_path, # <-- المسار الديناميكي الجديد
        valid_data_path='valid_16062025.csv',
        model_type='lstm',
        target_coin='btc'
    )
    # إضافة معلومات النموذج المستخدم إلى assets
    assets['model_info'] = {"loaded_model": model_filename}
    app.logger.info(f"--- [API] تم تحميل الأصول بنجاح باستخدام نموذج: {model_filename} ---")

except Exception as e:
    app.logger.error(f"FATAL: Could not load model assets on startup: {e}")
    assets = None # للدلالة على أن التحميل فشل


# --- نقاط النهاية (Endpoints) ---

@app.route('/health', methods=['GET'])
def health_check():
    """نقطة نهاية للتحقق من صحة الخدمة."""
    if assets is None:
        return jsonify({"status": "unhealthy", "message": "Model assets failed to load."}), 503
    return jsonify({"status": "ok", "message": "Service is running and assets are loaded."}), 200

@app.route('/info', methods=['GET'])
def model_info():
    """إرجاع معلومات عن النموذج المستخدم حالياً."""
    if assets and 'model_info' in assets:
        return jsonify(assets['model_info']), 200
    else:
        return jsonify({"error": "Information not available", "message": "Model assets might not be loaded."}), 503

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """نقطة النهاية الرئيسية لعمل التنبؤ."""
    if assets is None:
        abort(503, description="The prediction service is not ready. Model assets failed to load.")
        
    if not request.is_json:
        abort(400, description="Invalid request format. Expecting a JSON body.")

    data = request.get_json()

    try:
        prediction = make_prediction(assets=assets, input_data=data)
        return jsonify({"prediction": prediction})

    except ValueError as e:
        # التقاط اخطاء البيانات المرسلة بشكل واضح
        abort(400, description=str(e))
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction: {e}")
        abort(500)


# --- معالجات الأخطاء (Error Handlers) ---
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad Request", "message": error.description or "Invalid data received."}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found", "message": "This resource does not exist."}), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred on our end."}), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({"error": "Service Unavailable", "message": error.description or "The service is not ready to handle requests."}), 503


# --- تشغيل التطبيق ---
if __name__ == '__main__':
    # هذا السطر للتشغيل المحلي فقط
    app.run(debug=True, port=5000)
