import streamlit as st
import pandas as pd
import numpy as np
from ctgan import CTGAN
from ydata_profiling import ProfileReport
from faker import Faker
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import io

# Page config
st.set_page_config(page_title="CTGAN Synthetic Data Generator", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None

# Title
st.title("🔬 CTGAN Synthetic Data Generator")
st.markdown("Generate privacy-preserving synthetic data from your CSV files using Conditional Tabular GAN")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.session_state.data = data
    
    st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
    
    # Display basic info
    with st.expander("📊 View Original Data", expanded=False):
        st.dataframe(data.head(20))
        st.write("**Data Types:**")
        st.write(data.dtypes)
        st.write("**Basic Statistics:**")
        st.write(data.describe())
    
    # Anonymization options
    st.sidebar.subheader("🔒 Anonymization")
    anonymize_name = st.sidebar.checkbox("Anonymize 'Name' column (if exists)", value=True)
    
    # Identifier removal
    st.sidebar.subheader("🗑️ Remove Identifiers")
    id_keywords_input = st.sidebar.text_input(
        "Identifier keywords (comma-separated)", 
        value="id,ssn,passport,email,phone,roll no,contact,mobile,address"
    )
    id_keywords = [kw.strip().lower() for kw in id_keywords_input.split(',')]
    
    # Process data
    data_processed = data.copy()
    
    # Anonymize name column
    if anonymize_name and 'Name' in data_processed.columns:
        fake = Faker()
        data_processed['Name'] = [fake.first_name() for _ in range(len(data_processed))]
        st.info("ℹ️ 'Name' column has been anonymized")
    
    # Detect and remove identifiers
    possible_ids = [
        c for c in data_processed.columns
        if any(key in c.lower() for key in id_keywords)
    ]
    
    if possible_ids:
        st.warning(f"⚠️ Detected potential identifier columns: {', '.join(possible_ids)}")
        remove_ids = st.sidebar.checkbox("Remove detected identifiers", value=True)
        if remove_ids:
            data_processed = data_processed.drop(columns=possible_ids, errors='ignore')
            st.success(f"✅ Removed columns: {', '.join(possible_ids)}")
    
    # Detect categorical and numerical columns
    categorical_cols = data_processed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = data_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    st.sidebar.subheader("📋 Column Types")
    st.sidebar.write(f"**Categorical:** {len(categorical_cols)}")
    st.sidebar.write(f"**Numerical:** {len(numerical_cols)}")
    
    with st.expander("🔍 View Detected Column Types"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Categorical Columns:**")
            st.write(categorical_cols if categorical_cols else "None")
        with col2:
            st.write("**Numerical Columns:**")
            st.write(numerical_cols if numerical_cols else "None")
    
    # Handle missing values
    for c in categorical_cols:
        data_processed[c] = data_processed[c].astype('category')
    
    missing_count = data_processed.isna().sum().sum()
    if missing_count > 0:
        st.warning(f"⚠️ Found {missing_count} missing values. Applying imputation...")
        
        for c in numerical_cols:
            if data_processed[c].isna().any():
                data_processed[c] = data_processed[c].fillna(data_processed[c].median())
        
        for c in categorical_cols:
            if data_processed[c].isna().any():
                data_processed[c] = data_processed[c].cat.add_categories(['__MISSING__']).fillna('__MISSING__')
        
        st.success("✅ Missing values handled")
    
    # CTGAN Configuration
    st.sidebar.subheader("🧠 CTGAN Parameters")
    epochs = st.sidebar.slider("Training Epochs", min_value=50, max_value=500, value=200, step=50)
    num_samples = st.sidebar.number_input("Number of Synthetic Rows", min_value=10, max_value=100000, value=len(data_processed))
    
    # Train CTGAN
    if st.button("🚀 Generate Synthetic Data", type="primary"):
        with st.spinner("Training CTGAN model... This may take several minutes."):
            try:
                # Train model
                ctgan = CTGAN(epochs=epochs, verbose=False)
                ctgan.fit(data_processed, categorical_cols)
                
                # Generate synthetic data
                synthetic_data = ctgan.sample(int(num_samples))
                st.session_state.synthetic_data = synthetic_data
                
                st.success("✅ Synthetic data generated successfully!")
                
                # Display synthetic data
                st.subheader("📊 Synthetic Data Preview")
                st.dataframe(synthetic_data.head(20))
                
                # Download button
                csv = synthetic_data.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Synthetic Data (CSV)",
                    data=csv,
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")
    
    # Analysis section
    if st.session_state.synthetic_data is not None:
        st.divider()
        st.header("📈 Data Quality Analysis")
        
        synthetic_data = st.session_state.synthetic_data
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistical Comparison", "🔗 Correlation Analysis", "🔒 Privacy Check"])
        
        with tab1:
            if numerical_cols:
                st.subheader("Statistical Summary Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Data**")
                    st.dataframe(data_processed[numerical_cols].describe().T)
                
                with col2:
                    st.write("**Synthetic Data**")
                    st.dataframe(synthetic_data[numerical_cols].describe().T)
            else:
                st.info("No numerical columns for statistical comparison")
        
        with tab2:
            if len(numerical_cols) > 1:
                st.subheader("Correlation Matrix Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Data Correlation**")
                    st.dataframe(data_processed[numerical_cols].corr())
                
                with col2:
                    st.write("**Synthetic Data Correlation**")
                    st.dataframe(synthetic_data[numerical_cols].corr())
            else:
                st.info("Need at least 2 numerical columns for correlation analysis")
        
        with tab3:
            st.subheader("Privacy Metrics")
            
            with st.spinner("Computing privacy metrics..."):
                try:
                    # Convert to matrix
                    def to_matrix(real_df, synthetic_df, cat_cols, num_cols):
                        all_cat_data = pd.concat([real_df[cat_cols], synthetic_df[cat_cols]], axis=0) if cat_cols else pd.DataFrame()
                        
                        if not all_cat_data.empty:
                            enc = OneHotEncoder(handle_unknown='ignore')
                            enc.fit(all_cat_data)
                            X_real_cat = enc.transform(real_df[cat_cols]).toarray()
                            X_syn_cat = enc.transform(synthetic_df[cat_cols]).toarray()
                        else:
                            X_real_cat = np.empty((len(real_df), 0))
                            X_syn_cat = np.empty((len(synthetic_df), 0))
                        
                        X_real_num = real_df[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(real_df), 0))
                        X_syn_num = synthetic_df[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(synthetic_df), 0))
                        
                        X_real = np.hstack([X_real_num, X_real_cat])
                        X_syn = np.hstack([X_syn_num, X_syn_cat])
                        
                        return X_real, X_syn
                    
                    X_real, X_syn = to_matrix(data_processed, synthetic_data, categorical_cols, numerical_cols)
                    
                    # Nearest neighbor distance
                    nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_real)
                    dists, _ = nn.kneighbors(X_syn, n_neighbors=1)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Min NN Distance", f"{np.min(dists):.4f}")
                    with col2:
                        st.metric("Median NN Distance", f"{np.median(dists):.4f}")
                    with col3:
                        st.metric("Max NN Distance", f"{np.max(dists):.4f}")
                    
                    # Exact duplicates
                    exact_overlap = pd.merge(
                        data_processed.reset_index(drop=True),
                        synthetic_data.reset_index(drop=True),
                        how='inner'
                    ).shape[0]
                    
                    st.metric("Exact Overlapping Rows", exact_overlap)
                    
                    if exact_overlap == 0 and np.min(dists) > 0:
                        st.success("✅ No exact duplicates found. Synthetic data appears privacy-preserving.")
                    elif exact_overlap > 0:
                        st.warning(f"⚠️ Found {exact_overlap} exact duplicates between real and synthetic data.")
                    else:
                        st.info("ℹ️ Privacy metrics computed successfully")
                    
                except Exception as e:
                    st.error(f"❌ Error computing privacy metrics: {str(e)}")
        
        # Generate profiling reports
        st.divider()
        if st.button("📋 Generate Detailed Profile Reports (HTML)", type="secondary"):
            with st.spinner("Generating profile reports... This may take a while."):
                try:
                    profile_real = ProfileReport(data_processed, title='Profile: Real Data', minimal=True)
                    profile_synth = ProfileReport(synthetic_data, title='Profile: Synthetic Data', minimal=True)
                    
                    # Save to buffer
                    real_html = profile_real.to_html()
                    synth_html = profile_synth.to_html()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="⬇️ Download Real Data Profile",
                            data=real_html,
                            file_name="profile_real.html",
                            mime="text/html"
                        )
                    
                    with col2:
                        st.download_button(
                            label="⬇️ Download Synthetic Data Profile",
                            data=synth_html,
                            file_name="profile_synthetic.html",
                            mime="text/html"
                        )
                    
                    st.success("✅ Profile reports generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating profiles: {str(e)}")

else:
    st.info("👆 Please upload a CSV file to get started")
    
    st.markdown("""
    ### 📖 How to Use:
    1. Upload your cleaned CSV dataset
    2. Configure anonymization and identifier removal settings in the sidebar
    3. Adjust CTGAN parameters (epochs, number of samples)
    4. Click "Generate Synthetic Data" to train the model
    5. Download the synthetic dataset and analyze quality metrics
    
    ### 🎯 Features:
    - Automatic detection of categorical and numerical columns
    - Name anonymization using Faker library
    - Identifier column removal
    - Missing value imputation
    - Privacy metrics (nearest neighbor distance, exact duplicates)
    - Statistical and correlation analysis
    - Detailed profiling reports
    """)
