import streamlit as st
import pandas as pd
import numpy as np
from ctgan import CTGAN
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from io import BytesIO

st.set_page_config(page_title="CTGAN Synthetic Data Generator", layout="wide")
st.title("📊 CTGAN Synthetic Data Generator")
st.write("Upload ANY CSV → Clean → Fix Missing → Generate Synthetic Data")

# -------------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------------
file = st.file_uploader("Upload CSV file", type=["csv"])
if file is None:
    st.stop()

df_raw = pd.read_csv(file)
st.subheader("Original Data")
st.dataframe(df_raw.head())

# -------------------------------------------------------------
# STEP 1 — Remove Identifier Columns Automatically
# -------------------------------------------------------------
id_keywords = ['id','name','email','phone','mobile','contact','reg','roll','aadhar','address','passport']
id_cols = [c for c in df_raw.columns if any(k in c.lower() for k in id_keywords)]

df = df_raw.drop(columns=id_cols, errors="ignore")
if id_cols:
    st.warning(f"Dropped identifier-like columns: {id_cols}")

# -------------------------------------------------------------
# STEP 2 — Type Detection
# -------------------------------------------------------------
categorical_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# -------------------------------------------------------------
# STEP 3 — Drop High Cardinality Categorical Columns
# -------------------------------------------------------------
high_card_cols = [c for c in categorical_cols if df[c].nunique() > len(df) * 0.5]
df = df.drop(columns=high_card_cols, errors="ignore")

if high_card_cols:
    st.warning(f"Dropped high-cardinality columns (CTGAN cannot learn them): {high_card_cols}")

# Refresh categorical cols
categorical_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.write("Categorical columns:", categorical_cols)
st.write("Numeric columns:", numeric_cols)

# -------------------------------------------------------------
# STEP 4 — Fix Missing Values (Guaranteed No-NaN Mode)
# -------------------------------------------------------------
st.subheader("Fixing Missing Values (Guaranteed no NaNs)")

# 1. Fill numeric NaNs with median
if numeric_cols:
    imp_num = SimpleImputer(strategy="median")
    df[numeric_cols] = imp_num.fit_transform(df[numeric_cols])

# 2. Fill categorical NaNs with __MISSING__
if categorical_cols:
    for c in categorical_cols:
        df[c] = df[c].astype("category")
        df[c] = df[c].fillna("__MISSING__")

# FINAL SAFETY NET — remove ANY remaining nulls
df = df.fillna("__MISSING__")

# Ensure NO NaN remains
if df.isna().sum().sum() == 0:
    st.success("All missing values removed successfully — dataset is CTGAN-safe.")
else:
    st.error("ERROR: NaNs still exist. Something is wrong.")
    st.stop()

# -------------------------------------------------------------
# STEP 5 — Train CTGAN
# -------------------------------------------------------------
st.subheader("Train CTGAN Model")
epochs = st.slider("Epochs", 50, 300, 150)

if st.button("Train CTGAN"):
    with st.spinner("Training CTGAN... please wait"):
        try:
            ctgan = CTGAN(epochs=epochs, verbose=True)
            ctgan.fit(df, categorical_cols)
        except Exception as e:
            st.error("CTGAN training FAILED:")
            st.error(str(e))
            st.stop()

    st.success("CTGAN Training Complete!")

    # ---------------------------------------------------------
    # STEP 6 — Generate Synthetic Data
    # ---------------------------------------------------------
    synthetic = ctgan.sample(len(df))

    st.subheader("Synthetic Data Preview")
    st.dataframe(synthetic.head())

    # Download button
    buf = BytesIO()
    synthetic.to_csv(buf, index=False)
    st.download_button("Download Synthetic CSV", buf.getvalue(),
                       "synthetic_data.csv", "text/csv")

    # ---------------------------------------------------------
    # STEP 7 — Privacy Check (Optional)
    # ---------------------------------------------------------
    st.subheader("Privacy Check (Nearest Neighbor Distance)")

    def to_matrix(real, syn, cat_cols, num_cols):
        # Encode categoricals
        if cat_cols:
            enc = OneHotEncoder(handle_unknown='ignore')
            all_cat = pd.concat([real[cat_cols], syn[cat_cols]], axis=0)
            enc.fit(all_cat)
            Xr_cat = enc.transform(real[cat_cols]).toarray()
            Xs_cat = enc.transform(syn[cat_cols]).toarray()
        else:
            Xr_cat = np.zeros((len(real), 0))
            Xs_cat = np.zeros((len(syn), 0))

        # Numeric
        Xr_num = real[num_cols].to_numpy() if num_cols else np.zeros((len(real), 0))
        Xs_num = syn[num_cols].to_numpy() if num_cols else np.zeros((len(syn), 0))

        return np.hstack([Xr_num, Xr_cat]), np.hstack([Xs_num, Xs_cat])

    X_real, X_syn = to_matrix(df, synthetic, categorical_cols, numeric_cols)

    nn = NearestNeighbors(n_neighbors=1).fit(X_real)
    dists, _ = nn.kneighbors(X_syn)

    st.write("Min distance:", float(np.min(dists)))
    st.write("Median distance:", float(np.median(dists)))
    st.write("Max distance:", float(np.max(dists)))

    # Exact row check
    exact = pd.merge(df.reset_index(drop=True),
                     synthetic.reset_index(drop=True),
                     how="inner").shape[0]

    st.write("Exact overlapping rows:", int(exact))
