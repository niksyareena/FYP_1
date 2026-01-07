"""
Run the full data cleaning pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.pipeline.pipeline import DataCleaningPipeline


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


def main():
    #set pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    #load dataset
    print("Loading Adult dataset...")
    df = load_adult_dataset()
    
    #initialize pipeline
    pipeline = DataCleaningPipeline(output_dir='data/output')
    
    #configure pipeline
    config = {
        'profiling': True,
        'format_correction': {
            'normalize_strings': True,
            'standardize_dates': True,
            'correct_types': True,
            'string_case': 'lower'
        },
        'duplicate_detection': True,  
    }
    
    #run pipeline
    df_cleaned = pipeline.run(df, dataset_name='adult', config=config)
    
    #save cleaned dataset
    df_cleaned.to_csv('data/output/adult_cleaned.csv', index=False)
    print(f"\n✅ Pipeline completed successfully!")
    print(f"   Cleaned dataset saved to: data/output/adult_cleaned.csv")
    print(f"   Final shape: {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns")


if __name__ == "__main__":
    main()
