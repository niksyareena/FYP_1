"""
Test script for Data Profiler
"""

import pandas as pd
from src.profiling.data_profiler import DataProfiler

# Create sample data for testing
def create_sample_data():
    """Create a sample dataset for testing"""
    data = {
        'age': [25, 30, 35, None, 45, 50, 55, 60, 30, 35],
        'income': [30000, 45000, None, 60000, 75000, 80000, 90000, 95000, 45000, 60000],
        'education': ['HS-grad', 'Bachelors', 'Masters', 'HS-grad', 'Doctorate', 
                     'Bachelors', 'Masters', 'Doctorate', 'Bachelors', 'HS-grad'],
        'occupation': ['Sales', 'Tech', 'Tech', None, 'Management', 
                      'Tech', 'Management', 'Management', 'Tech', 'Sales'],
        'city': ['NY', 'LA', 'NY', 'LA', 'SF', 'NY', 'SF', 'LA', 'LA', 'NY']
    }
    return pd.DataFrame(data)

def main():
    print("\n" + "="*60)
    print("Testing Data Profiler")
    print("="*60 + "\n")
    
    # Create sample data
    df = create_sample_data()
    print("Sample dataset created:")
    print(df.head())
    print()
    
    # Create profiler
    profiler = DataProfiler()
    
    # Generate profile
    print("Generating profile...\n")
    profile = profiler.generate_profile(df)
    
    # Print summary
    profiler.print_summary()
    
    # Save report
    profiler.save_report('data/sample_profile_report.json')
    
    # Access specific information
    print("\n" + "="*60)
    print("Accessing Specific Information:")
    print("="*60)
    print(f"Total rows: {profile['dataset_level']['n_rows']}")
    print(f"Total columns: {profile['dataset_level']['n_columns']}")
    print(f"Columns with missing values: {len([k for k, v in profile['dataset_level']['missing_percentages'].items() if v > 0])}")
    
    if profile['numeric_columns']['columns']:
        print(f"\nNumeric column example (age):")
        age_stats = profile['numeric_columns']['columns'].get('age', {})
        for key, value in age_stats.items():
            if value is not None:
                print(f"  {key}: {value:.2f}")
    
    print("\n✓ Testing complete!")

if __name__ == '__main__':
    main()
