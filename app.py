import streamlit as st
import pandas as pd
import numpy as np
from ctgan import CTGAN
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer, KNNImputer
from io import BytesIO

# -----------------------------------------------------------
# STREAMLIT PAGE SETUP
# -----------------------------------------------------------
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")
st.title("📊 Synthetic Data Generator using CTGAN")
st.write("Upload ANY CSV dataset → Clean → Impute → Synthesize → Download results.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin.")
    st.stop()

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
data_raw = pd.read_csv(uploaded_file)
st.subheader("Original Data")
st.dataframe(data_raw.head())
st.write(f"Rows: {data_raw.shape[0]}, Columns: {data_raw.shape[1]}")

# -----------------------------------------------------------
# ID-LIKE COLUMNS REMOVAL
# -----------------------------------------------------------
st.subheader("🔒 Removing Identifier-like Columns")

id_keywords = ['id', 'ssn', 'passport', 'email', 'phone', 'mobile', 'contact',
               'roll', 'reg', 'aadhar', 'address', 'name']

id_cols = [c for c in data_raw.columns if any(k in c.lower() for k in id_keywords)]

data = data_raw.drop(columns=id_cols, errors='ignore')

st.info(f"Removed ID-like columns: {id_cols}")

# -----------------------------------------------------------
# MISSING VALUES SUMMARY
# -----------------------------------------------------------
st.subheader("📉 Missing Value Summary")

missing_df = pd.DataFrame({
    "missing_count": data.isna().sum(),
    "missing_pct": (data.isna().sum() / len(data)) * 100,
    "dtype": data.dtypes.astype(str)
}).sort_values("missing_pct", ascending=False)

st.dataframe(missing_df)

# -----------------------------------------------------------
# DROP HIGH MISSINGNESS COLUMNS
# -----------------------------------------------------------
st.subheader("🚫 Drop Columns With Too Many Missing Values")

threshold = st.slider("Drop columns with % missing greater than:", 0, 100, 80)
cols_drop_thresh = missing_df[missing_df["missing_pct"] > threshold].index.tolist()

if cols_drop_thresh:
    st.warning(f"Will drop columns: {cols_drop_thresh}")

data = data.drop(columns=cols_drop_thresh, errors='ignore')

# -----------------------------------------------------------
# TYPE DETECTION
# -----------------------------------------------------------
categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

st.write("Detected categorical columns:", categorical_cols)
st.write("Detected numerical columns:", numerical_cols)

# -----------------------------------------------------------
# HIGH CARDINALITY DETECTION (fix for Name/Address/etc)
# -----------------------------------------------------------
st.subheader("🚫 Auto-remove High-Cardinality Columns (CTGAN cannot learn them)")

high_cardinality_cols = [c for c in categorical_cols if data[c].nunique() > (0.5 * len(data))]

if high_cardinality_cols:
    st.warning(f"Dropping high-cardinality columns: {high_cardinality_cols}")

data = data.drop(columns=high_cardinality_cols, errors='ignore')

# Recompute categorical columns
categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# -----------------------------------------------------------
# MISSING VALUE IMPUTATION
# -----------------------------------------------------------
st.subheader("🧼 Missing Value Handling")

num_strategy = st.selectbox("Numeric Imputation Strategy",
                            ["median", "mean", "knn", "interpolate", "drop_rows"])

cat_strategy = st.selectbox("Categorical Imputation Strategy",
                            ["__MISSING__", "mode", "drop_rows"])

add_indicators = st.checkbox("Add missing-indicator columns", value=True)

# Option: drop rows
if num_strategy == "drop_rows" or cat_strategy == "drop_rows":
    before = len(data)
    data = data.dropna()
    st.success(f"Dropped rows with missing values. {before} → {len(data)} rows left.")

# Add missing flags
if add_indicators:
    for col in data.columns:
        if data[col].isna().any():
            data[f"{col}__missing_flag"] = data[col].isna().astype(int)

# Numeric imputation
if numerical_cols and num_strategy != "drop_rows":
    try:
        if num_strategy in ["median", "mean"]:
            imp = SimpleImputer(strategy=num_strategy)
            data[numerical_cols] = imp.fit_transform(data[numerical_cols])
        elif num_strategy == "knn":
            imp = KNNImputer(n_neighbors=5)
            data[numerical_cols] = imp.fit_transform(data[numerical_cols])
        elif num_strategy == "interpolate":
            data[numerical_cols] = data[numerical_cols].interpolate(limit_direction="both")
            data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
        st.success("Numeric missing values imputed successfully.")
    except Exception as e:
        st.error(f"Numeric imputation error: {e}")

# Categorical imputation
if categorical_cols and cat_strategy != "drop_rows":
    for col in categorical_cols:
        data[col] = data[col].astype("category")
        if data[col].isna().any():
            if cat_strategy == "__MISSING__":
                data[col] = data[col].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")
            else:
                mode_val = data[col].mode(dropna=True)[0]
                data[col] = data[col].fillna(mode_val)

st.success("Missing value handling complete.")

# -----------------------------------------------------------
# FINAL VALIDATION FOR CTGAN
# -----------------------------------------------------------
st.subheader("✔ Final Validation Before CTGAN")

categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

st.write("Final Categorical:", categorical_cols)
st.write("Final Numeric:", numerical_cols)

# CTGAN cannot train if no numeric + no categorical
if len(categorical_cols) == 0 and len(numerical_cols) == 0:
    st.error("No usable columns left for CTGAN. Upload a different dataset.")
    st.stop()

# -----------------------------------------------------------
# TRAIN CTGAN
# -----------------------------------------------------------
st.subheader("🤖 Train CTGAN")

epochs = st.slider("Epochs", 50, 500, 200)

if st.button("Train Model"):
    with st.spinner("Training CTGAN... this may take a few minutes"):
        try:
            ctgan = CTGAN(epochs=epochs, verbose=True)
            ctgan.fit(data, categorical_cols)
            st.success("CTGAN trained successfully!")
        except Exception as e:
            st.error(f"CTGAN training failed: {e}")
            st.stop()

    # -----------------------------------------------------------
    # GENERATE SYNTHETIC DATA
    # -----------------------------------------------------------
    synthetic = ctgan.sample(len(data))

    st.subheader("🧪 Synthetic Data Preview")
    st.dataframe(synthetic.head())

    # Download
    buf = BytesIO()
    synthetic.to_csv(buf, index=False)
    st.download_button("Download Synthetic CSV", buf.getvalue(),
                       file_name="synthetic_data.csv", mime="text/csv")

    # -----------------------------------------------------------
    # PRIVACY CHECK
    # -----------------------------------------------------------
    st.subheader("🔐 Privacy Check")

    def to_matrix(real_df, syn_df, cat_cols, num_cols):
        if cat_cols:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(pd.concat([real_df[cat_cols], syn_df[cat_cols]], axis=0))

            X_real_cat = enc.transform(real_df[cat_cols]).toarray()
            X_syn_cat = enc.transform(syn_df[cat_cols]).toarray()
        else:
            X_real_cat = np.zeros((len(real_df), 0))
            X_syn_cat = np.zeros((len(syn_df), 0))

        X_real_num = real_df[num_cols].to_numpy() if num_cols else np.zeros((len(real_df), 0))
        X_syn_num = syn_df[num_cols].to_numpy() if num_cols else np.zeros((len(syn_df), 0))

        return np.hstack([X_real_num, X_real_cat]), np.hstack([X_syn_num, X_syn_cat])

    X_real, X_syn = to_matrix(data, synthetic, categorical_cols, numerical_cols)
    nn = NearestNeighbors(n_neighbors=1).fit(X_real)
    dists, _ = nn.kneighbors(X_syn)

    st.write("Min Distance:", float(np.min(dists)))
    st.write("Median Distance:", float(np.median(dists)))
    st.write("Max Distance:", float(np.max(dists)))

    exact_overlap = pd.merge(data.reset_index(drop=True),
                             synthetic.reset_index(drop=True),
                             how="inner").shape[0]

    st.write("Exact Row Overlaps:", exact_overlap)
