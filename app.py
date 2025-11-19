import streamlit as st
import pandas as pd
import numpy as np
from ctgan import CTGAN
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from io import BytesIO

# ----------------------------------------------
# PAGE SETUP
# ----------------------------------------------
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")
st.title("📊 Synthetic Data Generator using CTGAN")
st.write("Upload ANY dataset (CSV) and generate high-quality synthetic data.")

# ----------------------------------------------
# FILE UPLOAD
# ----------------------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Original Data Preview")
    st.dataframe(data.head())

    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    # ----------------------------------------------
    # REMOVE ID-LIKE COLUMNS
    # ----------------------------------------------
    id_keywords = ['id','ssn','passport','email','phone','mobile','contact',
                   'roll','reg','aadhar','address','name']
    possible_ids = [c for c in data.columns if any(k in c.lower() for k in id_keywords)]

    data_clean = data.drop(columns=possible_ids, errors='ignore')

    st.success(f"Removed ID-like columns: {possible_ids}")

    # ----------------------------------------------
    # TYPE DETECTION
    # ----------------------------------------------
    categorical_cols = data_clean.select_dtypes(include=['object','category','bool']).columns.tolist()
    numerical_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()

    st.write("Categorical Columns:", categorical_cols)
    st.write("Numerical Columns:", numerical_cols)

    # ----------------------------------------------
    # HANDLE MISSING VALUES
    # ----------------------------------------------
    for c in numerical_cols:
        if data_clean[c].isna().any():
            data_clean[c] = data_clean[c].fillna(data_clean[c].median())

    for c in categorical_cols:
        data_clean[c] = data_clean[c].astype('category')
        if data_clean[c].isna().any():
            data_clean[c] = data_clean[c].cat.add_categories(['__MISSING__']).fillna('__MISSING__')

    st.info("Missing values cleaned.")

    # ----------------------------------------------
    # TRAIN CTGAN
    # ----------------------------------------------
    st.subheader("🤖 Training CTGAN Model")
    epochs = st.slider("Training Epochs", 50, 500, 200)

    if st.button("Train Model"):
        with st.spinner("Training CTGAN... may take a few minutes"):
            ctgan = CTGAN(epochs=epochs, verbose=True)
            ctgan.fit(data_clean, categorical_cols)

        st.success("Model Trained Successfully!")

        # ----------------------------------------------
        # GENERATE SYNTHETIC DATA
        # ----------------------------------------------
        num_rows = len(data_clean)
        synthetic_data = ctgan.sample(num_rows)

        st.subheader("🧪 Synthetic Data Preview")
        st.dataframe(synthetic_data.head())

        # ----------------------------------------------
        # DOWNLOAD SYNTHETIC DATA
        # ----------------------------------------------
        csv_buffer = BytesIO()
        synthetic_data.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Synthetic Data",
            data=csv_buffer.getvalue(),
            file_name="synthetic_data.csv",
            mime="text/csv"
        )

        # ----------------------------------------------
        # GENERATE PROFILES
        # ----------------------------------------------
        st.subheader("📈 Profiling Reports")

        if st.button("Generate Profiling Reports"):
            with st.spinner("Generating Reports..."):
                profile_real = ProfileReport(data_clean, title="Real Data", minimal=True)
                profile_synth = ProfileReport(synthetic_data, title="Synthetic Data", minimal=True)

                profile_real.to_file("real_profile.html")
                profile_synth.to_file("synthetic_profile.html")

            st.success("Reports Generated!")

            st.download_button(
                "⬇️ Download Real Data Profile",
                data=open("real_profile.html","rb").read(),
                file_name="real_profile.html"
            )

            st.download_button(
                "⬇️ Download Synthetic Data Profile",
                data=open("synthetic_profile.html","rb").read(),
                file_name="synthetic_profile.html"
            )

        # ----------------------------------------------
        # PRIVACY CHECK
        # ----------------------------------------------
        st.subheader("🔒 Privacy Check")

        def to_matrix(real_df, synth_df, cat_cols, num_cols):
            all_cat = pd.concat([real_df[cat_cols], synth_df[cat_cols]], axis=0)
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(all_cat)

            X_real_cat = enc.transform(real_df[cat_cols]).toarray() if cat_cols else np.zeros((len(real_df),0))
            X_syn_cat = enc.transform(synth_df[cat_cols]).toarray() if cat_cols else np.zeros((len(synth_df),0))

            X_real_num = real_df[num_cols].to_numpy() if num_cols else np.zeros((len(real_df),0))
            X_syn_num = synth_df[num_cols].to_numpy() if num_cols else np.zeros((len(synthetic_data),0))

            return np.hstack([X_real_num, X_real_cat]), np.hstack([X_syn_num, X_syn_cat])

        X_real, X_syn = to_matrix(data_clean, synthetic_data, categorical_cols, numerical_cols)

        nn = NearestNeighbors(n_neighbors=1).fit(X_real)
        dists, _ = nn.kneighbors(X_syn)

        st.write("Nearest-neighbour distance statistics:")
        st.write("Min:", np.min(dists))
        st.write("Median:", np.median(dists))
        st.write("Max:", np.max(dists))

        exact_overlap = pd.merge(
            data_clean.reset_index(drop=True),
            synthetic_data.reset_index(drop=True),
            how='inner'
        ).shape[0]

        st.write("Exact overlapping rows:", exact_overlap)

