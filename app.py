import streamlit as st
import pandas as pd
import numpy as np
from ctgan import CTGAN
from ydata_profiling import ProfileReport
from faker import Faker
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import io

# Critical fix for PyArrow compatibility
def fix_nullable_dtypes(df):
    """Convert nullable integer/boolean types to standard types for PyArrow compatibility"""
    if df is None or df.empty:
        return df
    
    df = df.copy()
    for col in df.columns:
        dtype_name = str(df[col].dtype)
        
        # Handle nullable integer types
        if dtype_name in ['Int64', 'Int32', 'Int16', 'Int8', 'UInt64', 'UInt32', 'UInt16', 'UInt8']:
            if df[col].isna().any():
                df[col] = df[col].astype('float64')
            else:
                try:
                    df[col] = df[col].astype('int64')
                except:
                    df[col] = df[col].astype('float64')
        
        # Handle boolean type
        elif dtype_name == 'boolean':
            df[col] = df[col].astype('bool')
        
        # Handle string type
        elif dtype_name == 'string':
            df[col] = df[col].astype('object')
    
    return df

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
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        
        st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
        
        # Display basic info with dtype fix
        with st.expander("📊 View Original Data", expanded=False):
            st.dataframe(fix_nullable_dtypes(data.head(20)), use_container_width=True)
            st.write("**Data Types:**")
            st.text(str(data.dtypes))
            st.write("**Basic Statistics:**")
            st.dataframe(fix_nullable_dtypes(data.describe()), use_container_width=True)
        
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
        if anonymize_name:
            name_cols = [col for col in data_processed.columns if 'name' in col.lower()]
            if name_cols:
                fake = Faker()
                for col in name_cols:
                    data_processed[col] = [fake.first_name() for _ in range(len(data_processed))]
                st.info(f"ℹ️ Anonymized columns: {', '.join(name_cols)}")
        
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
                    
                    # CRITICAL: Convert nullable dtypes immediately after generation
                    synthetic_data = fix_nullable_dtypes(synthetic_data)
                    st.session_state.synthetic_data = synthetic_data
                    
                    st.success("✅ Synthetic data generated successfully!")
                    
                    # Display synthetic data
                    st.subheader("📊 Synthetic Data Preview")
                    st.dataframe(synthetic_data.head(20), use_container_width=True)
                    
                    # Download button
                    csv = synthetic_data.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Synthetic Data (CSV)",
                        data=csv,
                        file_name="synthetic_data.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during generation: {str(e)}")
                    st.exception(e)
        
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
                        real_stats = data_processed[numerical_cols].describe().T
                        st.dataframe(fix_nullable_dtypes(real_stats), use_container_width=True)
                    
                    with col2:
                        st.write("**Synthetic Data**")
                        synth_stats = synthetic_data[numerical_cols].describe().T
                        st.dataframe(fix_nullable_dtypes(synth_stats), use_container_width=True)
                else:
                    st.info("No numerical columns for statistical comparison")
            
            with tab2:
                if len(numerical_cols) > 1:
                    st.subheader("Correlation Matrix Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Original Data Correlation**")
                        real_corr = data_processed[numerical_cols].corr()
                        st.dataframe(fix_nullable_dtypes(real_corr), use_container_width=True)
                    
                    with col2:
                        st.write("**Synthetic Data Correlation**")
                        synth_corr = synthetic_data[numerical_cols].corr()
                        st.dataframe(fix_nullable_dtypes(synth_corr), use_container_width=True)
                else:
                    st.info("Need at least 2 numerical columns for correlation analysis")
            
            with tab3:
                st.subheader("Privacy Metrics")
                
                with st.spinner("Computing privacy metrics..."):
                    try:
                        # Convert to matrix
                        def to_matrix(real_df, synthetic_df, cat_cols, num_cols):
                            # Fix dtypes first
                            real_df = fix_nullable_dtypes(real_df)
                            synthetic_df = fix_nullable_dtypes(synthetic_df)
                            
                            all_cat_data = pd.concat([real_df[cat_cols], synthetic_df[cat_cols]], axis=0) if cat_cols else pd.DataFrame()
                            
                            if not all_cat_data.empty:
                                enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                                enc.fit(all_cat_data)
                                X_real_cat = enc.transform(real_df[cat_cols])
                                X_syn_cat = enc.transform(synthetic_df[cat_cols])
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
                        
                        # Exact duplicates - fix dtypes before merge
                        real_fixed = fix_nullable_dtypes(data_processed.reset_index(drop=True))
                        synth_fixed = fix_nullable_dtypes(synthetic_data.reset_index(drop=True))
                        
                        exact_overlap = pd.merge(
                            real_fixed,
                            synth_fixed,
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
                        st.exception(e)
            
            # Generate profiling reports
            st.divider()
            if st.button("📋 Generate Detailed Profile Reports (HTML)", type="secondary"):
                with st.spinner("Generating profile reports... This may take a while."):
                    try:
                        # Fix dtypes before profiling
                        profile_real = ProfileReport(
                            fix_nullable_dtypes(data_processed), 
                            title='Profile: Real Data', 
                            minimal=True,
                            explorative=False
                        )
                        profile_synth = ProfileReport(
                            fix_nullable_dtypes(synthetic_data), 
                            title='Profile: Synthetic Data', 
                            minimal=True,
                            explorative=False
                        )
                        
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
                        st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

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
    
    ### ⚠️ Notes:
    - Larger datasets require more training epochs
    - Training time increases with dataset complexity
    - Ensure your data is cleaned before upload
    """)
