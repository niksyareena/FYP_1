"""
Data Quality Pipeline - Streamlit Interface
Interactive demo for data cleaning and quality assessment
"""

import streamlit as st
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Page configuration
st.set_page_config(
    page_title="Data Quality Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Lazy imports - only load when page loads, not on every interaction
    from utils.visualizations import plot_missing_values, plot_data_types
    
    # Header
    st.markdown('<p class="main-header">(system name)</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Dataset upload
    st.sidebar.subheader("1. Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file to analyze and clean"
    )
    
    df = None
    dataset_name = None
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name.replace('.csv', '')
        st.sidebar.success(f"✅ Loaded: {dataset_name}")
    
    if df is not None:
        # Sampling option for large datasets
        st.sidebar.subheader("2. Sampling Options")
        if len(df) > 10000:
            st.sidebar.info(f"📊 Dataset has {len(df):,} records")
            use_sample = st.sidebar.checkbox(
                "Use sample for faster demo (recommended)", 
                value=True,
                help="Sampling speeds up processing for demonstration. Full evaluation uses complete datasets."
            )
            
            if use_sample:
                sample_size = st.sidebar.slider(
                    "Sample size", 
                    min_value=1000, 
                    max_value=min(10000, len(df)), 
                    value=5000,
                    step=1000
                )
                df = df.sample(n=sample_size, random_state=42)
                st.sidebar.success(f"✅ Using {len(df):,} records for demo")
        
        # Execution mode
        st.sidebar.subheader("3. Execution Mode")
        execution_mode = st.sidebar.radio(
            "Choose execution mode",
            ["Run Full Pipeline", "Run Individual Module"],
            help="Full Pipeline runs all modules sequentially. Individual Module runs only selected module."
        )
        
        # Module configuration based on execution mode
        st.sidebar.subheader("4. Module Configuration")
        
        # Initialize selected_module with default value
        selected_module = "Data Profiling"
        
        if execution_mode == "Run Individual Module":
            selected_module = st.sidebar.selectbox(
                "Select Module",
                ["Data Profiling", "Format Correction", "Duplicate Detection"]
            )
            
            # Initialize default values
            normalize_strings = True
            standardize_dates = False
            correct_types = True
            string_case = "lower"
            fuzzy_threshold = 0.85
            run_fuzzy = False
            
            # Module-specific options
            if selected_module == "Format Correction":
                with st.sidebar.expander("Format Correction Options"):
                    normalize_strings = st.checkbox("Normalize Strings", value=True)
                    standardize_dates = st.checkbox("Standardize Dates", value=False)
                    correct_types = st.checkbox("Correct Types", value=True)
                    string_case = st.selectbox("String Case", ["lower", "upper", "title", "unchanged"])
            
            elif selected_module == "Duplicate Detection":
                with st.sidebar.expander("Duplicate Detection Options"):
                    fuzzy_threshold = st.slider("Fuzzy Match Threshold", 0.7, 1.0, 0.85, 0.05)
                    run_fuzzy = st.checkbox(
                        "Include Fuzzy Duplicates (slower)", 
                        value=False,
                        help="Fuzzy duplicate detection may take several minutes on large datasets"
                    )
        
        else:  # Full Pipeline
            st.sidebar.write("All modules will be executed:")
            st.sidebar.write("✓ Data Profiling")
            st.sidebar.write("✓ Format Correction")
            st.sidebar.write("✓ Duplicate Detection")
            
            with st.sidebar.expander("Pipeline Options"):
                normalize_strings = st.checkbox("Normalize Strings", value=True)
                standardize_dates = st.checkbox("Standardize Dates", value=False)
                correct_types = st.checkbox("Correct Types", value=True)
                string_case = st.selectbox("String Case", ["lower", "upper", "title", "unchanged"])
                fuzzy_threshold = st.slider("Fuzzy Match Threshold", 0.7, 1.0, 0.85, 0.05)
                run_fuzzy = st.checkbox(
                    "Include Fuzzy Duplicates (slower)", 
                    value=False,
                    help="Fuzzy duplicate detection may take several minutes on large datasets"
                )
        
        # Run button
        st.sidebar.markdown("---")
        if execution_mode == "Run Individual Module":
            run_pipeline = st.sidebar.button(f"🚀 Run {selected_module}", type="primary", use_container_width=True)
        else:
            run_pipeline = st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["📋 Overview", "🔍 Results", "📁 Data"])
        
        with tab1:
            st.header("Dataset Loaded")
            st.success(f"✅ {dataset_name} loaded successfully with {len(df):,} rows and {len(df.columns)} columns")
            st.info("👈 Configure and run a module from the sidebar to see results")
        
        # Run pipeline or module
        if run_pipeline:
            if execution_mode == "Run Individual Module":
                # Run individual module
                with st.spinner(f"🔄 Running {selected_module}..."):
                    df_original = df.copy()
                    df_result = df.copy()
                    
                    try:
                        if selected_module == "Data Profiling":
                            from src.profiling.data_profiler import DataProfiler
                            profiler = DataProfiler()
                            profile = profiler.generate_profile(df)
                            
                            st.session_state['module_name'] = 'Data Profiling'
                            st.session_state['profile'] = profile
                            st.session_state['profiler'] = profiler
                        
                        elif selected_module == "Format Correction":
                            from src.format_correction.format_corrector import FormatCorrector
                            corrector = FormatCorrector()
                            format_config = {
                                'normalize_strings': normalize_strings,
                                'standardize_dates': standardize_dates,
                                'correct_types': correct_types,
                                'string_case': string_case
                            }
                            df_result = corrector.correct_formats(df, format_config)
                            
                            st.session_state['module_name'] = 'Format Correction'
                            st.session_state['corrector'] = corrector
                        
                        elif selected_module == "Duplicate Detection":
                            from src.duplicates.duplicate_detector import DuplicateDetector
                            detector = DuplicateDetector(fuzzy_threshold=fuzzy_threshold)
                            
                            # Detect exact duplicates
                            detector.detect_duplicates(df)
                            
                            # Remove them if found
                            if detector.duplicates_found > 0:
                                df_result = df.drop_duplicates(keep='first').reset_index(drop=True)
                            else:
                                df_result = df.copy()
                            
                            # Fuzzy duplicates if enabled
                            if run_fuzzy:
                                detector.detect_fuzzy_duplicates(df_result)
                                if detector.fuzzy_duplicate_pairs:
                                    df_result = detector.remove_fuzzy_duplicates(df_result, interactive=False)
                            
                            st.session_state['module_name'] = 'Duplicate Detection'
                            st.session_state['detector'] = detector
                        
                        # Store results
                        st.session_state['df_original'] = df_original
                        st.session_state['df_cleaned'] = df_result
                        st.session_state['execution_mode'] = 'individual'
                        st.session_state['pipeline_complete'] = True
                        
                        st.success(f"✅ {selected_module} completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error running {selected_module}: {str(e)}")
                        st.exception(e)
            
            else:  # Run full pipeline
                with st.spinner("🔄 Running full data quality pipeline..."):
                    # Build configuration
                    config = {
                        'profiling': True,
                        'format_correction': {
                            'normalize_strings': normalize_strings,
                            'standardize_dates': standardize_dates,
                            'correct_types': correct_types,
                            'string_case': string_case
                        },
                        'duplicate_detection': True,
                        'fuzzy_duplicates': run_fuzzy,
                        'fuzzy_threshold': fuzzy_threshold
                    }
                    
                    # Initialize pipeline
                    from src.pipeline.pipeline import DataCleaningPipeline
                    pipeline = DataCleaningPipeline(output_dir='data/output')
                    
                    # Store original for comparison
                    df_original = df.copy()
                    
                    # Run pipeline
                    try:
                        df_cleaned = pipeline.run(df, dataset_name=dataset_name or "dataset", config=config)
                        
                        # Store in session state
                        st.session_state['df_original'] = df_original
                        st.session_state['df_cleaned'] = df_cleaned
                        st.session_state['pipeline'] = pipeline
                        st.session_state['execution_mode'] = 'full'
                        st.session_state['pipeline_complete'] = True
                        
                        st.success("✅ Full pipeline completed successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error running pipeline: {str(e)}")
                        st.exception(e)
        
        # Display results if pipeline has run
        if st.session_state.get('pipeline_complete', False):
            df_original = st.session_state['df_original']
            df_cleaned = st.session_state['df_cleaned']
            execution_mode = st.session_state.get('execution_mode', 'full')
            
            with tab2:
                if execution_mode == 'individual':
                    module_name = st.session_state.get('module_name', 'Module')
                    st.header(f"{module_name} Results")
                    
                    # Module-specific results
                    if module_name == "Data Profiling":
                        profiler = st.session_state.get('profiler')
                        profile = st.session_state.get('profile')
                        
                        if profile:
                            # Dataset-Level Statistics
                            st.subheader("📊 Dataset-Level Statistics")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Rows", f"{profile['dataset_level']['n_rows']:,}")
                            with col2:
                                st.metric("Columns", profile['dataset_level']['n_columns'])
                            with col3:
                                st.metric("Missing Values", f"{sum(profile['dataset_level']['missing_counts'].values()):,}")
                            with col4:
                                st.metric("Duplicates", f"{profile['dataset_level']['n_duplicates']:,}")
                            with col5:
                                memory_mb = profile['dataset_level']['memory_usage'] / (1024 * 1024)
                                st.metric("Memory", f"{memory_mb:.2f} MB")
                            
                            st.markdown("---")
                            
                            # Data Types Distribution
                            st.subheader("📋 Data Types")
                            dtypes_dict = profile['dataset_level']['dtypes']
                            dtypes_df = pd.DataFrame(list(dtypes_dict.items()), columns=['Column', 'Type'])
                            st.dataframe(dtypes_df, use_container_width=True)
                            
                            st.markdown("---")
                            
                            # Missing Values per Column
                            st.subheader("❓ Missing Values by Column")
                            missing_dict = profile['dataset_level']['missing_counts']
                            if any(missing_dict.values()):
                                missing_df = pd.DataFrame(list(missing_dict.items()), columns=['Column', 'Missing Count'])
                                missing_df['Missing %'] = (missing_df['Missing Count'] / profile['dataset_level']['n_rows'] * 100).round(2)
                                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                                st.dataframe(missing_df, use_container_width=True)
                            else:
                                st.success("✅ No missing values found in dataset")
                            
                            st.markdown("---")
                            
                            # Numeric columns
                            if profile.get('numeric'):
                                st.subheader("📊 Numeric Columns Statistics")
                                numeric_df = pd.DataFrame(profile['numeric']).T
                                st.dataframe(numeric_df, use_container_width=True)
                                
                                # Correlation matrix if multiple numeric columns
                                if len(profile['numeric']) > 1:
                                    st.markdown("---")
                                    st.subheader("🔗 Correlation Matrix")
                                    # Get correlation data from first numeric column
                                    first_col = list(profile['numeric'].keys())[0]
                                    if 'correlations' in profile['numeric'][first_col]:
                                        corr_dict = {}
                                        for col, stats in profile['numeric'].items():
                                            if 'correlations' in stats:
                                                corr_dict[col] = stats['correlations']
                                        if corr_dict:
                                            corr_df = pd.DataFrame(corr_dict)
                                            st.dataframe(corr_df.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1), use_container_width=True)
                            
                            st.markdown("---")
                            
                            # Categorical columns
                            if profile.get('categorical'):
                                st.subheader("📑 Categorical Columns Statistics")
                                cat_df = pd.DataFrame(profile['categorical']).T
                                st.dataframe(cat_df, use_container_width=True)
                    
                    elif module_name == "Format Correction":
                        corrector = st.session_state.get('corrector')
                        
                        if corrector and corrector.corrections_log:
                            st.subheader("🔧 Format Corrections Applied")
                            corrections_df = corrector.get_corrections_summary()
                            st.dataframe(corrections_df, use_container_width=True)
                            
                            st.markdown("---")
                            st.subheader("Summary Statistics")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Corrections", len(corrector.corrections_log))
                            with col2:
                                st.metric("Rows Affected", corrections_df['rows_affected'].sum())
                        else:
                            st.info("No format corrections were needed.")
                    
                    elif module_name == "Duplicate Detection":
                        detector = st.session_state.get('detector')
                        
                        if detector:
                            st.subheader("👥 Duplicate Detection Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Exact Duplicates Found", detector.duplicates_found)
                            with col2:
                                st.metric("Fuzzy Duplicate Pairs", len(detector.fuzzy_duplicate_pairs))
                            with col3:
                                duplicates_removed = len(df_original) - len(df_cleaned)
                                st.metric("Total Removed", duplicates_removed)
                            
                            if detector.duplicates_log:
                                st.markdown("---")
                                st.subheader("Detection Log")
                                for log in detector.duplicates_log:
                                    st.json(log)
                
                else:  # Full pipeline
                    pipeline = st.session_state['pipeline']
                    st.header("Pipeline Results")
                    
                    # Overall metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Rows Processed",
                            f"{len(df_original):,}",
                            delta=f"{len(df_cleaned) - len(df_original):,}" if len(df_cleaned) != len(df_original) else "0"
                        )
                    
                    with col2:
                        missing_before = df_original.isnull().sum().sum()
                        missing_after = df_cleaned.isnull().sum().sum()
                        st.metric(
                            "Missing Values Filled",
                            f"{missing_before - missing_after:,}",
                            delta=f"-{((missing_before-missing_after)/max(missing_before,1)*100):.1f}%" if missing_before > missing_after else "0%"
                        )
                    
                    with col3:
                        duplicates = df_original.duplicated().sum() - df_cleaned.duplicated().sum()
                        st.metric(
                            "Duplicates Removed",
                            f"{duplicates:,}"
                        )
                    
                    st.markdown("---")
                    
                    # Module-specific results
                    if pipeline.format_corrector.corrections_log:
                        st.subheader("🔧 Format Corrections")
                        corrections_df = pipeline.format_corrector.get_corrections_summary()
                        st.dataframe(corrections_df, use_container_width=True)
                    
                    if pipeline.duplicate_detector.duplicates_log:
                        st.subheader("👥 Duplicate Detection")
                        for log in pipeline.duplicate_detector.duplicates_log:
                            st.json(log)
                    
                    # Before/After comparison
                    if pipeline.profile_before and pipeline.profile_after:
                        from utils.ui_components import display_profile_comparison
                        st.markdown("---")
                        display_profile_comparison(pipeline.profile_before, pipeline.profile_after)
            
            with tab3:
                st.header("Data Quality Visualizations")
                
                # Missing values comparison
                st.subheader("Missing Values: Before vs After")
                fig = plot_missing_values(df_original, df_cleaned)
                st.plotly_chart(fig, use_container_width=True)
                
                # Data type distribution
                st.subheader("Column Data Types")
                fig = plot_data_types(df_cleaned)
                st.plotly_chart(fig, use_container_width=True)
                
                # Download section
                st.markdown("---")
                st.subheader("Download Cleaned Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_original = df_original.to_csv(index=False)
                    st.download_button(
                        "📥 Download Original CSV",
                        csv_original,
                        file_name=f"{dataset_name or 'dataset'}_original.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    csv_cleaned = df_cleaned.to_csv(index=False)
                    st.download_button(
                        "📥 Download Cleaned CSV",
                        csv_cleaned,
                        file_name=f"{dataset_name or 'dataset'}_cleaned.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    else:
        st.info("Please upload a dataset from the sidebar to begin")


if __name__ == "__main__":
    main()
