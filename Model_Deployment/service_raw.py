# service_raw.py
# FastAPI service that accepts RAW transactions, applies your saved preprocessor,
# builds a (T x F) window per request, and returns proba + decision.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from joblib import load as joblib_load
try:
    import keras          
except ImportError:
    from tensorflow import keras

import numpy as np
import pandas as pd
import json, os
from datetime import datetime

# Config & artifact paths
ART_DIR = "artifacts_fraud_detec"

MODEL_PATH   = os.path.join(ART_DIR, "lstm_fraud_model.keras")
PREPROC_PATH = os.path.join(ART_DIR, "preprocessor.joblib")
WIN_PATH     = os.path.join(ART_DIR, "windowing.json")
THRESH_PATH  = os.path.join(ART_DIR, "threshold.json")

# Load artifacts ONCE at startup
model = keras.models.load_model(MODEL_PATH)
pre   = joblib_load(PREPROC_PATH)

with open(WIN_PATH) as f:
    win = json.load(f)
T        = int(win["timesteps"])

ID_COL   = win["id_col"]
TIME_COL = win["time_col"]

with open(THRESH_PATH) as f:
    t_star = float(json.load(f)["t_star"])

# Raw features that your preprocessor expects (before encoding)
FEATURES_RAW = ["amt", "trans_hour", "time_since_last", "last_amt", "category"]

# Determine the actual encoded feature count by probing the preprocessor
def _get_encoded_feature_count():
    """Probe the preprocessor to determine output feature count."""
    probe_data = pd.DataFrame({
        "amt": [100.0],
        "trans_hour": [12],
        "time_since_last": [3600.0],
        "last_amt": [50.0],
        "category": ["gas_transport"]  # Use a valid category
    })
    encoded = pre.transform(probe_data)
    return encoded.shape[1]

F = _get_encoded_feature_count()
print(f"Loaded model expecting T={T}, F={F} (encoded features)")

# Request schema
class Txn(BaseModel):
    # Keys for windowing
    cc_num: str | int
    trans_date_trans_time: datetime
    
    # Raw features (5 features before encoding)
    amt: float
    trans_hour: int
    time_since_last: float
    last_amt: float
    category: str

class PredictBody(BaseModel):
    transactions: List[Txn]

# Helpers: cleaning, preprocessing, windowing
def clean_category_series(s: pd.Series) -> pd.Series:
    """Normalize category strings."""
    s = s.astype("string").str.strip().str.lower()
    # Replace empty or pad tokens with a valid default or NaN
    s = s.replace("", pd.NA).replace("__PAD__", pd.NA)
    return s

def preprocess_raw_block(df_raw: pd.DataFrame) -> np.ndarray:
    """
    Apply the saved preprocessor to raw features.
    Returns: (n_rows, F_encoded) array
    """
    df = df_raw.copy()
    
    # Ensure correct dtypes
    df["amt"] = df["amt"].astype("float64")
    df["trans_hour"] = df["trans_hour"].astype("int64")
    df["time_since_last"] = df["time_since_last"].astype("float64")
    df["last_amt"] = df["last_amt"].astype("float64")
    df["category"] = clean_category_series(df["category"])
    
    # Apply preprocessor (scales numerics, one-hot encodes category)
    X_enc = pre.transform(df[FEATURES_RAW])
    return np.asarray(X_enc, dtype=np.float32)

def build_window_for_single_id(df_enc: pd.DataFrame, id_col: str, time_col: str, T: int, F: int) -> np.ndarray:
    """
    Build a single (T, F) window from encoded transactions for one ID.
    Takes the last T transactions sorted by time, pads with zeros if needed.
    
    Args:
        df_enc: DataFrame with encoded features + id and time columns
        id_col: Name of ID column
        time_col: Name of timestamp column
        T: Number of timesteps
        F: Number of encoded features
    
    Returns:
        Array of shape (T, F)
    """
    # Sort by time
    df_sorted = df_enc.sort_values(time_col).reset_index(drop=True)
    
    # Extract feature columns (everything except id and time)
    feature_cols = [c for c in df_sorted.columns if c not in [id_col, time_col]]
    arr = df_sorted[feature_cols].to_numpy()
    
    n_rows, n_feats = arr.shape
    
    # Validate feature count
    if n_feats != F:
        raise ValueError(f"Feature count mismatch. Expected {F}, got {n_feats}")
    
    # Take last T rows or pad if fewer
    if n_rows >= T:
        window = arr[-T:]
    else:
        # Pad with zeros at the beginning
        pad = np.zeros((T - n_rows, F), dtype=arr.dtype)
        window = np.vstack([pad, arr])
    
    return window  # shape (T, F)

# FastAPI app + endpoints
app = FastAPI(title="Fraud LSTM Service (raw → preproc → window)", version="v2")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timesteps": T,
        "n_features_encoded": F,
        "threshold": t_star,
        "raw_features": FEATURES_RAW
    }

@app.post("/predict")
def predict(body: PredictBody):
    """
    Accepts transactions with raw features, encodes them, builds a window, and predicts fraud probability.
    """
    try:
        # Convert to DataFrame
        data = [t.dict() for t in body.transactions]
        df = pd.DataFrame(data)
        
        # Validate we have exactly one cc_num (single-ID request)
        unique_ids = df["cc_num"].unique()
        if len(unique_ids) != 1:
            raise ValueError(f"Provide transactions for exactly one cc_num per request. Found {len(unique_ids)} IDs.")
        
        owner_id = unique_ids[0]
        
        # Convert timestamp column
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        
        # 1) Extract raw features and encode them
        df_raw_features = df[FEATURES_RAW].copy()
        X_encoded = preprocess_raw_block(df_raw_features)  # (n, F) where F=17
        
        # 2) Create DataFrame with encoded features
        # Get feature names from preprocessor if available
        try:
            enc_feature_names = pre.get_feature_names_out()
        except AttributeError:
            enc_feature_names = [f"feature_{i}" for i in range(F)]
        
        df_encoded = pd.DataFrame(X_encoded, columns=enc_feature_names, index=df.index)
        
        # 3) Add back the ID and timestamp columns for windowing
        df_encoded["cc_num"] = df["cc_num"].values
        df_encoded["trans_date_trans_time"] = df["trans_date_trans_time"].values
        
        # 4) Build window (T, F)
        window = build_window_for_single_id(
            df_encoded, 
            id_col="cc_num", 
            time_col="trans_date_trans_time",
            T=T, 
            F=F
        )
        
        # 5) Reshape for LSTM: (1, T, F)
        X_3D = window.reshape(1, T, F)
        
        # 6) Predict
        prob_fraud = float(model.predict(X_3D, verbose=0).ravel()[0])
        decision = int(prob_fraud >= t_star)
        
        return {
            "cc_num": str(owner_id),
            "n_transactions": len(df),
            "proba_fraud": prob_fraud,
            "decision": decision,
            "threshold": t_star,
            "window_shape": f"({T}, {F})"
        }
    
    except Exception as e:
        import traceback
        raise HTTPException(status_code=400, detail=f"{str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)




