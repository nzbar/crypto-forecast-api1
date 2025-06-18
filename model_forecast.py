# model_forecast.py

import pandas as pd
import json
import torch
from pretrain.gru import GRU
from pretrain.lstm import LSTM
from statsmodels.tsa.holtwinters import Holt
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_prediction_assets(config_path, features_path, model_path, model_type, valid_data_path, target_coin):
    """
    تحميل جميع الأصول اللازمة للتنبؤ مرة واحدة.
    هذه الدالة تُستدعى مرة واحدة فقط عند بدء تشغيل الخادم.
    """
    print("Loading prediction assets...")
    
    # تحميل الإعدادات والميزات
    with open(config_path) as f: config = json.load(f)
    with open(features_path) as f: features = json.load(f)['features']
    
    # تحميل النموذج
    model_class = LSTM if model_type.lower() == 'lstm' else GRU
    model = model_class(
        n_features=len(features),
        hidden_units=config['hidden_units'],
        n_layers=config['n_layers'],
        lr=config['learning_rate']
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # تحميل بيانات التحقق لتهيئة المحجمات (Scalers)
    valid_df = pd.read_csv(valid_data_path, index_col='Date', parse_dates=True)
    target_col_name = f"{target_coin.lower()}_avg_ohlc"
    
    # تهيئة المحجمات
    main_scaler = MinMaxScaler().fit(valid_df[features + [target_col_name]])
    target_only_scaler = MinMaxScaler().fit(valid_df[[target_col_name]])
    
    print("Assets loaded successfully.")
    
    # إرجاع قاموس يحتوي على كل الأصول المحملة
    return {
        "model": model,
        "config": config,
        "features": features,
        "main_scaler": main_scaler,
        "target_only_scaler": target_only_scaler,
        "valid_df": valid_df, # سنحتاجها للحصول على آخر تسلسل
        "target_col_name": target_col_name
    }

def make_prediction(assets, horizon=7):
    """
    تقوم بعملية التنبؤ باستخدام الأصول المحملة مسبقًا.
    هذه الدالة سريعة ويمكن استدعاؤها مع كل طلب API.
    """
    # استخراج الأصول من القاموس
    model = assets['model']
    valid_df = assets['valid_df']
    features = assets['features']
    config = assets['config']
    main_scaler = assets['main_scaler']
    target_only_scaler = assets['target_only_scaler']
    target_col_name = assets['target_col_name']

    # 1. توقع الميزات المستقبلية باستخدام Holt
    pred_set = pd.DataFrame()
    for feature in features:
        holt = Holt(valid_df[feature], initialization_method="estimated").fit()
        pred = holt.forecast(horizon)
        pred_set = pd.concat([pred_set, pred], axis=1)
    pred_set.columns = features

    # 2. الحصول على آخر تسلسل من البيانات
    sequence_length = config.get('sequence_length', 60)
    last_sequence_unscaled = valid_df[features].iloc[-sequence_length:]
    
    # 3. دمج التسلسل الأخير مع الميزات المتوقعة
    full_sequence_unscaled = pd.concat([last_sequence_unscaled, pred_set]).iloc[-sequence_length:]
    
    # 4. تحجيم (Scale) التسلسل
    full_sequence_scaled = main_scaler.transform(pd.concat([full_sequence_unscaled, pd.DataFrame(columns=[target_col_name])]))[:, :len(features)]
    
    # 5. إجراء التنبؤ
    with torch.no_grad():
        input_tensor = torch.tensor(full_sequence_scaled, dtype=torch.float).unsqueeze(0).to(DEVICE)
        y_hat = model(input_tensor)

    # 6. عكس التحجيم للحصول على القيمة الحقيقية
    prediction_scaled = y_hat.cpu().numpy().flatten()
    final_forecast = target_only_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    # 7. إرجاع النتيجة
    return float(final_forecast[0])