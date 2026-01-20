"""
Reusable UI components for Streamlit interface
"""

import streamlit as st
import pandas as pd


def display_dataframe_info(df, title):
    """Display basic dataframe information"""
    st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Duplicates", f"{df.duplicated().sum():,}")


def display_profile_comparison(profile_before, profile_after):
    """Display before/after profile comparison"""
    st.subheader("📊 Data Quality Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Before Cleaning")
        st.metric("Total Rows", f"{profile_before['dataset_level']['n_rows']:,}")
        st.metric("Missing Values", f"{sum(profile_before['dataset_level']['missing_counts'].values()):,}")
        st.metric("Duplicates", f"{profile_before['dataset_level']['n_duplicates']:,}")
    
    with col2:
        st.markdown("### After Cleaning")
        st.metric("Total Rows", f"{profile_after['dataset_level']['n_rows']:,}")
        st.metric("Missing Values", f"{sum(profile_after['dataset_level']['missing_counts'].values()):,}")
        st.metric("Duplicates", f"{profile_after['dataset_level']['n_duplicates']:,}")
