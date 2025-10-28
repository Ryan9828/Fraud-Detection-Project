# service_raw.py
# ----------------
# FastAPI service that accepts RAW transactions, applies your saved preprocessor,
# builds a (T x F) window per request, and returns proba + decision.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from joblib import load as joblib_load
try:
    import keras            # Keras 3 (standalone) – preferred for loading a Keras 3 model
except ImportError:
    from tensorflow import keras  # fallback if only tf.keras is available

import numpy as np
import pandas as pd
import json, os
from datetime import datetime

# ---------- 1) Config & artifact paths ----------
ART_DIR = "artifacts_fraud_detec"

MODEL_PATH   = os.path.join(ART_DIR, "lstm_fraud_model.keras")
PREPROC_PATH = os.path.join(ART_DIR, "preprocessor.joblib")
WIN_PATH     = os.path.join(ART_DIR, "windowing.json")
THRESH_PATH  = os.path.join(ART_DIR, "threshold.json")

# ---------- 2) Load artifacts ONCE at startup ----------
model = keras.models.load_model(MODEL_PATH)
pre   = joblib_load(PREPROC_PATH)

with open(WIN_PATH) as f:
    win = json.load(f)
T        = int(win["timesteps"])
# Don't use F from windowing.json yet - we'll calculate it
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

# ---------- 3) Request schema (validated by Pydantic) ----------
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

# ---------- 4) Helpers: cleaning, preprocessing, windowing ----------
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

# ---------- 5) FastAPI app + endpoints ----------
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
















#############################################################################################################
# # service_raw.py
# # ----------------
# # FastAPI service that accepts RAW transactions, applies your saved preprocessor,
# # builds a (T x F) window per request, and returns proba + decision.

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from typing import List
# from joblib import load as joblib_load
# #from tensorflow import keras
# try:
#     import keras            # Keras 3 (standalone) — preferred for loading a Keras 3 model
# except ImportError:
#     from tensorflow import keras  # fallback if only tf.keras is available


# import numpy as np
# import pandas as pd
# import json, os
# from datetime import datetime

# # ---------- 1) Config & artifact paths ----------
# ART_DIR = "artifacts_fraud_detec"

# MODEL_PATH   = os.path.join(ART_DIR, "lstm_fraud_model.keras")
# PREPROC_PATH = os.path.join(ART_DIR, "preprocessor.joblib")
# WIN_PATH     = os.path.join(ART_DIR, "windowing.json")
# THRESH_PATH  = os.path.join(ART_DIR, "threshold.json")

# # ---------- 2) Load artifacts ONCE at startup ----------
# model = keras.models.load_model(MODEL_PATH)
# pre   = joblib_load(PREPROC_PATH)

# with open(WIN_PATH) as f:
#     win = json.load(f)
# T        = int(win["timesteps"])
# F        = int(win["n_features"])     # encoded feature count
# ID_COL   = win["id_col"]
# TIME_COL = win["time_col"]

# with open(THRESH_PATH) as f:
#     t_star = float(json.load(f)["t_star"])

# # We’ll use the same raw column order you trained with:
# # (If you saved schema.json with "features_in_order", load that. Otherwise, set manually.)
# try:
#     with open(os.path.join(ART_DIR, "schema.json")) as f:
#         schema = json.load(f)
#     FEATURES_RAW = schema["features_in_order"]
#     DTYPES       = schema["dtypes"]
# except Exception:
#     # Fallback if you didn’t save schema.json
#     FEATURES_RAW = ["amt","trans_hour","time_since_last","last_amt","category"]
#     DTYPES = {"amt":"float64","trans_hour":"int64","time_since_last":"float64","last_amt":"float64","category":"string"}

# ######################################################################################
# def _probe_row_from_dtypes(dtypes: dict) -> pd.DataFrame:
#     # construct a single-row DataFrame with correct dtypes
#     sample = {}
#     for k, v in dtypes.items():
#         if k not in FEATURES_RAW:
#             continue
#         if v in ("float64", "float32"):
#             sample[k] = pd.Series([0.0], dtype=v)
#         elif v in ("int64", "int32"):
#             sample[k] = pd.Series([0], dtype=v)
#         elif v == "string":
#             # use a benign category token; your OneHot should ignore unknowns
#             sample[k] = pd.Series(["home"], dtype="string")
#         else:
#             sample[k] = pd.Series([""], dtype="string")
#     return pd.DataFrame(sample)

# _probe = _probe_row_from_dtypes(DTYPES).reindex(columns=FEATURES_RAW, fill_value=np.nan)
# F_enc = pre.transform(_probe).shape[1]
# F = int(F_enc)  # override the value loaded from windowing.json
# ##################################################################################
# # ---------- 3) Request schema (validated by Pydantic) ----------
# class Txn(BaseModel):
#     # use aliases so the JSON matches your training columns
#     id: str | int = Field(..., alias=ID_COL)
#     ts: datetime  = Field(..., alias=TIME_COL)

#     # Raw features the preprocessor expects (adjust if yours differ)
#     amt: float
#     trans_hour: int
#     time_since_last: float
#     last_amt: float
#     category: str

# class PredictBody(BaseModel):
#     transactions: List[Txn]

# # ---------- 4) Helpers: cleaning, preprocessing, windowing ----------
# PAD_TOKEN = "__PAD__"   # if you used this in your window builder

# def clean_category_series(s: pd.Series) -> pd.Series:
#     # normalize case/whitespace; turn PAD into NaN so OneHotEncoder(handle_unknown="ignore") stays quiet
#     s = s.astype("string").str.strip().str.lower()
#     return s.replace(PAD_TOKEN, pd.NA)

# def preprocess_raw_block(df_raw: pd.DataFrame) -> np.ndarray:
#     # coerce dtypes exactly as training
#     dtype_map = {k: ("string" if v=="string" else v) for k,v in DTYPES.items() if k in df_raw.columns}
#     df_raw = df_raw.copy()
#     for col, typ in dtype_map.items():
#         df_raw[col] = df_raw[col].astype(typ)

#     if "category" in df_raw.columns:
#         df_raw["category"] = clean_category_series(df_raw["category"])

#     X_enc = pre.transform(df_raw[FEATURES_RAW])     # -> 2D array (n_rows, F_enc)
#     return np.asarray(X_enc, dtype=float)

# def last_T_window(df_enc_with_keys: pd.DataFrame) -> np.ndarray:
#     # Sort by time and take the last T rows; left-pad zeros if fewer than T
#     g   = df_enc_with_keys.sort_values(TIME_COL)
#     arr = g.drop(columns=[ID_COL, TIME_COL]).to_numpy()
#     n, f = arr.shape
#     if f != F:
#         raise ValueError(f"Encoded feature size mismatch. Model expects F={F}, got {f}.")
#     if n >= T:
#         win = arr[-T:]
#     else:
#         pad = np.zeros((T - n, f), dtype=arr.dtype)
#         win = np.vstack([pad, arr])
#     return win  # shape (T, F)

# # ---------- 5) FastAPI app + endpoints ----------
# app = FastAPI(title="Fraud LSTM Service (raw → preproc → window)", version="v1")

# @app.get("/health")
# def health():
#     return {"status": "ok", "timesteps": T, "n_features": F, "threshold": t_star}

# @app.post("/predict")
# def predict(body: PredictBody):
#     try:
#         # Build a DataFrame from incoming transactions
#         df = pd.DataFrame([t.dict(by_alias=True) for t in body.transactions])

#         # Validate single id per request (simple demo). You can batch by grouping later.
#         owners = df[ID_COL].unique()
#         if len(owners) != 1:
#             raise ValueError(f"Provide transactions for exactly one {ID_COL} per request.")
#         owner = owners[0]

#         # Preprocess (encode) the raw features block
#         df_raw = df[FEATURES_RAW].copy()
#         X_enc  = preprocess_raw_block(df_raw)
#         enc_cols = getattr(pre, "get_feature_names_out", lambda: [f"f{i}" for i in range(X_enc.shape[1])])()
#         df_enc = pd.DataFrame(X_enc, columns=enc_cols, index=df.index)

#         # Attach keys back for windowing
#         df_enc[ID_COL]   = df[ID_COL].values
#         df_enc[TIME_COL] = pd.to_datetime(df[TIME_COL].values)

#         # Build last-T window and predict
#         win_tf = last_T_window(df_enc)        # (T, F)
#         X3D    = win_tf.reshape(1, T, F)      # (1, T, F)
#         p      = float(model.predict(X3D, verbose=0).ravel()[0])
#         return {"id": owner, "proba_fraud": p, "decision": int(p >= t_star), "threshold": t_star}

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
