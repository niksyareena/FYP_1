"""
Data loading utilities for different datasets
"""

import pandas as pd
import streamlit as st


@st.cache_data
def load_adult_dataset():
    """Load UCI Adult dataset"""
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    df = pd.read_csv(
        'datasets/adult/adult.data',
        names=column_names,
        header=None,
        na_values=' ?',
        skipinitialspace=True
    )
    return df


@st.cache_data
def load_bank_dataset():
    """Load Bank Marketing dataset"""
    df = pd.read_csv('datasets/bank-additional/bank-additional-full.csv', sep=';')
    return df


def load_dataset(dataset_name):
    """Load selected dataset"""
    if dataset_name == "Adult (UCI)":
        return load_adult_dataset(), "adult"
    elif dataset_name == "Bank Marketing":
        return load_bank_dataset(), "bank"
    else:
        return None, None
