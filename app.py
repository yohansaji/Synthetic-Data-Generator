# app.py
import streamlit as st
import pandas as pd
import numpy as np
from ctgan import CTGAN
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer, KNNImputer
from io import BytesIO

st.set_page_config(page_title="Synthetic Data Generator (robust missing handling)", layout="wide")
st.title("📊 Synthetic Data Generator — Missing Value Handling Improved")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV to get started.")
    st.stop()

# Load
data_raw = pd.read_csv(uploaded_file)
st.subheader("Original Data")
st.write(f"Rows: {data_raw.shape[0]}  Columns: {data_raw.shape[1]}")
st.dataframe(data_raw.head())

# Missing summary
st.subheader("Missing value summary")
miss_counts = data_raw.isna().sum()
miss_pct = (miss_counts / len(data_raw)) * 100
missing_df = pd.DataFrame({
    "missing_count": miss_counts,
    "missing_pct": miss_pct,
    "dtype": data_raw.dtypes.astype(str)
}).sort_values("missing_pct", ascending=False)
st.dataframe(missing_df)

# User controls: drop columns with too many missing values
st.write("### Column drop threshold")
col_thresh = st.slider("Drop columns with > % missing", 0, 100, 80)
cols_to_drop_by_thresh = missing_df[missing_df["missing_pct"] > col_thresh].index.tolist()
st.write(f"Columns that would be dropped (> {col_thresh}% missing): {cols_to_drop_by_thresh}")

# Option to drop selected columns manually
st.write("### Manually drop columns (optional)")
cols_selected = st.multiselect("Select additional columns to drop", options=list(data_raw.columns))
cols_to_drop = list(set(cols_to_drop_by_thresh + cols_selected))
if cols_to_drop:
    st.warning(f"These columns will be dropped: {cols_to_drop}")

# Choose strategy for handling remaining missing values
st.write("### Imputation strategies (remaining columns)")

num_strategy = st.selectbox("Numeric imputation strategy", options=["median", "mean", "knn", "interpolate", "drop_rows"])
cat_strategy = st.selectbox("Categorical imputation strategy", options=["__MISSING__", "mode", "drop_rows"])

# Add missing indicators?
add_indicators = st.checkbox("Add missing-indicator columns for features with missing values", value=True)

# Make a working copy and drop columns
data = data_raw.drop(columns=cols_to_drop, errors='ignore').copy()

# Recompute types
categorical_cols = data.select_dtypes(include=['object','category','bool']).columns.tolist()
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

st.write("Detected categorical columns:", categorical_cols)
st.write("Detected numerical columns:", numerical_cols)

# Option to coerce certain columns to categorical (helpful if codes exist)
with st.expander("Optionally convert numeric columns to categorical (e.g., codes)"):
    to_cat = st.multiselect("Select numeric columns to convert to category", options=numerical_cols)
    for c in to_cat:
        data[c] = data[c].astype("object")
    if to_cat:
        # refresh lists
        categorical_cols = data.select_dtypes(include=['object','category','bool']).columns.tolist()
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        st.write("Updated categorical columns:", categorical_cols)
        st.write("Updated numerical columns:", numerical_cols)

# ===== Missing-handling pipeline =====
st.subheader("Applying missing-value pipeline")

# 1) Optionally drop rows if user chose drop_rows for either strategy
drop_rows_if = (num_strategy == "drop_rows") or (cat_strategy == "drop_rows")
if drop_rows_if:
    before_rows = len(data)
    data = data.dropna()
    st.warning(f"Dropped rows with any missing values. Rows before: {before_rows} → after: {len(data)}")

# 2) Add missing indicators
if add_indicators:
    cols_with_missing = [c for c in data.columns if data[c].isna().any()]
    for c in cols_with_missing:
        data[f"{c}__missing_flag"] = data[c].isna().astype(int)
    st.info(f"Added missing-indicator columns for: {cols_with_missing}")

# 3) Numeric imputation
if numerical_cols and num_strategy != "drop_rows":
    if num_strategy in ["median", "mean"]:
        strategy = "median" if num_strategy == "median" else "mean"
        imp = SimpleImputer(strategy=strategy)
        try:
            data[numerical_cols] = imp.fit_transform(data[numerical_cols])
            st.success(f"Numeric columns imputed with {strategy}.")
        except Exception as e:
            st.error(f"Numeric imputation failed ({e}). Falling back to median.")
            imp = SimpleImputer(strategy="median")
            data[numerical_cols] = imp.fit_transform(data[numerical_cols])
    elif num_strategy == "knn":
        try:
            imp = KNNImputer(n_neighbors=5)
            data[numerical_cols] = imp.fit_transform(data[numerical_cols])
            st.success("Numeric columns imputed with KNNImputer (k=5).")
        except Exception as e:
            st.error(f"KNN imputer failed ({e}). Falling back to median.")
            imp = SimpleImputer(strategy="median")
            data[numerical_cols] = imp.fit_transform(data[numerical_cols])
    elif num_strategy == "interpolate":
        try:
            data[numerical_cols] = data[numerical_cols].interpolate(method='linear', limit_direction='both')
            # Fill any remaining with median
            remaining_na = data[numerical_cols].isna().sum().sum()
            if remaining_na > 0:
                st.warning(f"{remaining_na} numeric NAs remain after interpolate — filling with median.")
                data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
            else:
                st.success("Numeric columns interpolated.")
        except Exception as e:
            st.error(f"Interpolate failed ({e}). Filling numerics with median.")
            data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# 4) Categorical imputation
if categorical_cols and cat_strategy != "drop_rows":
    if cat_strategy == "__MISSING__":
        for c in categorical_cols:
            data[c] = data[c].astype("category")
            if data[c].isna().any():
                data[c] = data[c].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")
        st.success("Categorical missing values replaced with '__MISSING__' token.")
    elif cat_strategy == "mode":
        for c in categorical_cols:
            try:
                mode_val = data[c].mode(dropna=True)[0]
            except Exception:
                mode_val = "__MISSING__"
            data[c] = data[c].fillna(mode_val).astype("category")
        st.success("Categorical missing values filled with mode (most frequent).")

# Final check
remaining_nas = data.isna().sum().sum()
if remaining_nas == 0:
    st.success("No remaining missing values.")
else:
    st.error(f"Still {remaining_nas} missing values remain. Consider different strategies or drop rows/columns.")

# Recompute categorical/numeric lists for CTGAN
categorical_cols = data.select_dtypes(include=['object','category','bool']).columns.tolist()
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

st.write("Final categorical columns:", categorical_cols)
st.write("Final numerical columns:", numerical_cols)

# ---- ID-like column removal (keep earlier suggestion) ----
id_keywords = ['id','ssn','passport','email','phone','mobile','contact','roll','reg','aadhar','address','name']
possible_ids = [c for c in data.columns if any(k in c.lower() for k in id_keywords)]
# Remove only if they are not newly added missing-indicators (avoid removing flags)
possible_ids = [c for c in possible_ids if not c.endswith("__missing_flag")]
data = data.drop(columns=possible_ids, errors='ignore')
if possible_ids:
    st.info(f"Removed ID-like columns: {possible_ids}")

# ====== CTGAN training ======
st.subheader("CTGAN Training")
epochs = st.slider("Training Epochs", 50, 500, 200)
if st.button("Train CTGAN"):
    with st.spinner("Training CTGAN (this may take a while)..."):
        ctgan = CTGAN(epochs=epochs, verbose=True)
        # If no categorical cols, pass empty list
        try:
            ctgan.fit(data, categorical_cols)
            st.success("CTGAN trained successfully.")
        except Exception as e:
            st.error(f"CTGAN training failed: {e}")
            st.stop()

    # generate synthetic
    num_rows = len(data)
    synthetic_data = ctgan.sample(num_rows)
    st.subheader("Synthetic Data Preview")
    st.dataframe(synthetic_data.head())

    # downloads
    buf = BytesIO()
    synthetic_data.to_csv(buf, index=False)
    st.download_button("Download synthetic CSV", data=buf.getvalue(), file_name="synthetic_data.csv", mime="text/csv")

    # Profiles
    if st.button("Generate profiling reports"):
        with st.spinner("Generating profile reports..."):
            profile_real = ProfileReport(data, title="Real Data", minimal=True)
            profile_synth = ProfileReport(synthetic_data, title="Synthetic Data", minimal=True)
            profile_real.to_file("real_profile.html")
            profile_synth.to_file("synth_profile.html")
        st.success("Profiles generated.")
        st.download_button("Download real profile", data=open("real_profile.html","rb").read(), file_name="real_profile.html")
        st.download_button("Download synth profile", data=open("synth_profile.html","rb").read(), file_name="synth_profile.html")

    # privacy check
    st.subheader("Privacy check / nearest neighbour distances")
    def to_matrix(real_df, synthetic_df, cat_cols, num_cols):
        all_cat = pd.concat([real_df[cat_cols], synthetic_df[cat_cols]], axis=0) if cat_cols else pd.DataFrame()
        enc = OneHotEncoder(handle_unknown='ignore')
        if cat_cols:
            enc.fit(all_cat)
            X_real_cat = enc.transform(real_df[cat_cols]).toarray()
            X_syn_cat = enc.transform(synthetic_df[cat_cols]).toarray()
        else:
            X_real_cat = np.zeros((len(real_df),0))
            X_syn_cat = np.zeros((len(synthetic_df),0))

        X_real_num = real_df[num_cols].to_numpy() if num_cols else np.zeros((len(real_df),0))
        X_syn_num = synthetic_df[num_cols].to_numpy() if num_cols else np.zeros((len(synthetic_df),0))

        return np.hstack([X_real_num, X_real_cat]), np.hstack([X_syn_num, X_syn_cat])

    X_real, X_syn = to_matrix(data, synthetic_data, categorical_cols, numerical_cols)
    nn = NearestNeighbors(n_neighbors=1).fit(X_real)
    dists, _ = nn.kneighbors(X_syn)
    st.write("NN distances (min, median, max):", float(np.min(dists)), float(np.median(dists)), float(np.max(dists)))
    exact_overlap = pd.merge(data.reset_index(drop=True), synthetic_data.reset_index(drop=True), how='inner').shape[0]
    st.write("Exact overlapping rows:", int(exact_overlap))

else:
    st.info("Press 'Train CTGAN' to train model and generate synthetic data.")
