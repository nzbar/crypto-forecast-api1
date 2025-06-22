# train_worker.py

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, timedelta
import os

# استيراد دوالك ونماذجك
# ملاحظة: تأكد من أن هذه الملفات تحتوي على النسخ الأصلية المناسبة للتدريب
from data_pull import fetch_crypto_data_from_coingecko
from feature_engineering import create_features
from pretrain.lstm import LSTM # النسخة التي ترث من pl.LightningModule
from pretrain.gru import GRU   # النسخة التي ترث من pl.LightningModule

# --- الإعدادات ---
# هذا هو المسار إلى القرص الصلب الدائم الذي أنشأناه في Render
MODEL_OUTPUT_DIR = '/data/models' 
# قائمة العملات
COIN_LIST = [
    'btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'busd', 'ada', 
    'sol', 'doge', 'dot', 'dai', 'shib', 'trx', 'avax', 'uni', 
    'wbtc', 'leo', 'ltc'
]
DAYS_TO_FETCH = 365 # قد تحتاج لبيانات أكثر للتدريب
SEQUENCE_LENGTH = 60
TARGET_COIN = 'btc'

def prepare_dataloaders(features_df, target_col_name, batch_size):
    """
    تجهيز محملات البيانات للتدريب والتحقق.
    """
    # فصل البيانات إلى تدريب وتحقق (مثلاً، آخر 10% للتحقق)
    train_size = int(len(features_df) * 0.9)
    train_df, val_df = features_df[:train_size], features_df[train_size:]

    # ... هنا يجب وضع كود تجهيز البيانات وتحويلها إلى TensorDatasets ...
    # هذه الخطوة معقدة وتعتمد على كيفية بناءك للـ Dataset في الأصل.
    # كمثال مبسط جداً:
    features = [col for col in features_df.columns if col != target_col_name]

    X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[target_col_name].values, dtype=torch.float32)

    X_val = torch.tensor(val_df[features].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[target_col_name].values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(features)

def run_training_job():
    print("--- [WORKER] بدء مهمة التدريب المجدولة ---")

    # --- 1. جلب البيانات وهندسة الميزات ---
    print("[1/3] جلب ومعالجة البيانات...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    all_raw_dfs = []
    for coin in COIN_LIST:
        coin_df = fetch_crypto_data_from_coingecko(coin, start_date.strftime('%d-%m-%Y'), end_date.strftime('%d-%m-%Y'))
        if coin_df is not None:
            coin_df['Coin'] = coin.upper()
            all_raw_dfs.append(coin_df)

    raw_df = pd.concat(all_raw_dfs)
    raw_df.reset_index(inplace=True)
    features_df = create_features(raw_df)
    print("  - تم تجهيز بيانات الميزات بنجاح.")

    # --- 2. إعداد وتدريب النموذج ---
    print("[2/3] إعداد النموذج والبدء بالتدريب...")
    target_col = f"{TARGET_COIN.lower()}_avg_ohlc"
    train_loader, val_loader, n_features = prepare_dataloaders(features_df, target_col, batch_size=64)

    # إنشاء مجلد حفظ النماذج إذا لم يكن موجوداً
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # إعداد نقاط الحفظ والتوقف المبكر
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_OUTPUT_DIR,
        filename=f'model-{datetime.now().strftime("%Y%m%d")}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    # تهيئة النموذج (تأكد من أن هذا هو الكلاس الأصلي للتدريب)
    model = LSTM(n_features=n_features, lr=1e-4) # مثال

    # تهيئة المدرب
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='cpu',
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # بدء التدريب
    trainer.fit(model, train_loader, val_loader)

    # --- 3. الانتهاء ---
    print(f"[3/3] اكتمل التدريب! تم حفظ أفضل نموذج في: {checkpoint_callback.best_model_path}")
    print("\n✅ نجحت مهمة التدريب المجدولة!")
# train_worker.py -> inside run_training_job()

# ... (كود التدريب trainer.fit(...))

# --- 4. الإعلان عن النموذج الجديد ---
# التحقق من أن التدريب نجح وأن هناك نموذج جديد تم حفظه
if checkpoint_callback.best_model_path:
    best_model_filename = os.path.basename(checkpoint_callback.best_model_path)
    latest_model_info_path = os.path.join(MODEL_OUTPUT_DIR, 'latest.txt')

    print(f"[4/4] تحديث ملف 'latest.txt' ليشير إلى النموذج الجديد: {best_model_filename}")

    # كتابة اسم الملف الجديد في الملف الوسيط
    with open(latest_model_info_path, 'w') as f:
        f.write(best_model_filename)

    print("\n✅ نجحت مهمة التدريب المجدولة وتم تحديث مؤشر أحدث نموذج!")
else:
    print("\n❌ فشل التدريب أو لم يتم حفظ أي نموذج جديد.")


if __name__ == "__main__":
    run_training_job()