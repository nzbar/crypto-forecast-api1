# app.py

from flask import Flask, request, jsonify
from model_forecast import load_prediction_assets, make_prediction
import os

# --- الإعدادات الأساسية (تم تحديثها لتناسب هيكل ملفاتك) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. تحديث مسار ملف الإعدادات إلى الاسم الصحيح: config_nn.json
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config_nn.json')

# 2. مسار ملف الميزات صحيح، لا داعي للتغيير
FEATURES_PATH = os.path.join(BASE_DIR, 'config', 'features.json')

# 3. مسار النموذج صحيح بناءً على آخر تعديل
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lstm_16062025.pth')

# 4. تحديث مسار ملف التحقق: الملف موجود في المجلد الرئيسي ومختلف الاسم
VALID_DATA_PATH = os.path.join(BASE_DIR, 'valid_16062025.csv')

# 5. نوع النموذج صحيح بناءً على آخر تعديل
TARGET_COIN = 'btc' # الرجاء التأكد من أن هذا هو اسم العملة الصحيح الذي تستهدفه
MODEL_TYPE = 'lstm'

# --- تهيئة التطبيق وتحميل الأصول ---
app = Flask(__name__)

# تحميل الأصول مرة واحدة عند بدء تشغيل الخادم وتخزينها في متغير
try:
    prediction_assets = load_prediction_assets(
        config_path=CONFIG_PATH,
        features_path=FEATURES_PATH,
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        valid_data_path=VALID_DATA_PATH,
        target_coin=TARGET_COIN
    )
except FileNotFoundError as e:
    print(f"Error loading assets: {e}. Make sure all paths are correct.")
    prediction_assets = None # Handle case where loading fails
except Exception as e:
    print(f"An unexpected error occurred during asset loading: {e}")
    prediction_assets = None

# --- تحديد رابط الـ API ---
@app.route('/predict', methods=['GET'])
def handle_prediction():
    if prediction_assets is None:
        return jsonify({'error': 'Prediction assets could not be loaded. Check server logs.'}), 500

    try:
        forecasted_price = make_prediction(prediction_assets)
        
        return jsonify({
            'status': 'success',
            'predicted_price': f'{forecasted_price:.2f}'
        })

    except Exception as e:
        app.logger.error(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500

# --- تشغيل التطبيق ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)