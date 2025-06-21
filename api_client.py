# api_client.py (النسخة النهائية - تعالج كل العملات)
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# استيراد الدوال من ملفاتك
from data_pull import fetch_crypto_data_from_coingecko
from feature_engineering import create_features 

# --- إعدادات العميل ---
API_URL = "http://localhost:8000/predict"
SEQUENCE_LENGTH = 60
DAYS_TO_FETCH = 200

# === التعديل الأهم: قائمة بكل العملات التي يحتاجها النموذج ===
# أضف أو أزل من هذه القائمة لتطابق متطلبات نموذجك
COIN_LIST = [
    'btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'busd', 'ada', 
    'sol', 'doge', 'dot', 'dai', 'shib', 'trx', 'avax', 'uni', 
    'wbtc', 'leo', 'ltc'
]

# --- باقي الدوال تبقى كما هي ---
def prepare_payload(features_df: pd.DataFrame) -> dict:
    print(f"    - تحويل آخر {SEQUENCE_LENGTH} صف إلى الصيغة المطلوبة...")
    if len(features_df) < SEQUENCE_LENGTH:
        raise ValueError(f"البيانات غير كافية، نحتاج على الأقل {SEQUENCE_LENGTH} صفاً ولكن المتوفر {len(features_df)} صفاً فقط.")
    latest_sequence_df = features_df.tail(SEQUENCE_LENGTH).copy()
    if isinstance(latest_sequence_df.index, pd.DatetimeIndex):
        latest_sequence_df = latest_sequence_df.reset_index()
    for col in latest_sequence_df.columns:
        if pd.api.types.is_datetime64_any_dtype(latest_sequence_df[col]):
            latest_sequence_df[col] = latest_sequence_df[col].dt.strftime('%Y-%m-%d')
    sequence_as_list = latest_sequence_df.to_dict(orient='records')
    payload = {"sequence": sequence_as_list}
    return payload

def call_api(payload: dict) -> float:
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result.get('prediction')
    except requests.exceptions.HTTPError as http_err:
        print(f"!! خطأ في الـ API (HTTP Error): {http_err}")
        print(f"   تفاصيل الرد: {response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"!! خطأ في الاتصال بالـ API: {req_err}")
        return None

# === تعديل دالة main لتجلب بيانات كل العملات ===
def main():
    print("--- بدء عملية العميل (API Client) ---")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    end_date_str = end_date.strftime('%d-%m-%Y')
    start_date_str = start_date.strftime('%d-%m-%Y')
    
    print(f"\n[ الخطوة 1/4 ] جلب بيانات كل العملات من {start_date_str} إلى {end_date_str}...")
    
    all_raw_dfs = []
    # حلقة جديدة للمرور على كل العملات وجلب بياناتها
    for coin in COIN_LIST:
        print(f"    - جلب بيانات {coin.upper()}...")
        coin_df = fetch_crypto_data_from_coingecko(coin, start_date_str, end_date_str)
        if coin_df is not None:
            coin_df['Coin'] = coin.upper() # إضافة عمود باسم العملة
            all_raw_dfs.append(coin_df)

    if not all_raw_dfs:
        print("\n❌ فشلت عملية جلب البيانات لأي عملة، لن يتم إكمال العملية.")
        return

    # دمج كل بيانات العملات في DataFrame واحد خام
    raw_df = pd.concat(all_raw_dfs)
    raw_df.reset_index(inplace=True) # للتأكد من أن التاريخ عمود وليس فهرس
    print("    - تم دمج بيانات كل العملات بنجاح.")

    print("\n[ الخطوة 2/4 ] حساب الميزات المطلوبة لكل العملات...")
    # الآن نمرر الـ DataFrame المدمج الذي يحتوي على كل شيء
    features_df = create_features(raw_df) 
    print("    - تم حساب الميزات بنجاح.")
    
    print("\n[ الخطوة 3/4 ] تجهيز حمولة JSON للـ API...")
    payload = prepare_payload(features_df)
    print("    - تم تجهيز الحمولة بنجاح.")
    
    print("\n[ الخطوة 4/4 ] إرسال الطلب إلى الـ API والحصول على التنبؤ...")
    prediction = call_api(payload)
    
    print("\n--- انتهت العملية ---")
    if prediction is not None:
        print(f"\n✅ النتيجة النهائية: التنبؤ الذي تم الحصول عليه من الـ API هو: {prediction}")
    else:
        print("\n❌ فشلت عملية الحصول على التنبؤ.")


if __name__ == "__main__":
    main()