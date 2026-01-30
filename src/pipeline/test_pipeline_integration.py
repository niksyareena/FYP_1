"""


This script tests the integrated pipeline performance across modules:
1. Data Profiling (runs ONCE at the start - before any fixes)
2. Format Correction  
3. Duplicate Detection (Exact + Fuzzy)

Tracks execution time for each module and total pipeline time.

"""

import pandas as pd
import time
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.profiling.data_profiler import DataProfiler
from src.format_correction.format_corrector import FormatCorrector
from src.duplicates.duplicate_detector import DuplicateDetector


# CONFIGURATION 

DATASET = 'adult'
SAMPLE_SIZE = 2000  # Set to None for full dataset/ specify number for sampling
RANDOM_STATE = 42


# HELPER FUNCTIONS
def format_time(seconds):

    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} min {secs:.1f} sec"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} hr {int(minutes)} min"


def print_pipeline_header(title):
    print("\n" + "─" * 80)
    print(f"│ {title}")
    print("─" * 80)


def print_timing(label, seconds):
    print(f"  ⏱  {label}: {format_time(seconds)}")


def load_dataset(dataset_name, sample_size, random_state=42):
    """
    Load the specified dataset with optional sampling
    
    Args:
        dataset_name: 'adult', 'bank', or 'breast cancer'
        sample_size: Number of rows to sample (None = full dataset)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (dataframe, full_dataset_size)
    """
    print("\n" + "═" * 80)
    print(f"  LOADING DATASET: {dataset_name.upper()}")
    print("═" * 80)
    
    #dataset paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    if dataset_name == 'adult':
        filepath = os.path.join(project_root, 'datasets/adult/adult.csv')
        df = pd.read_csv(filepath)
    elif dataset_name == 'breast cancer':
        filepath = os.path.join(project_root, 'datasets/breast+cancer/breast-cancer.csv')
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    full_size = len(df)
    
    print(f"  Full dataset: {full_size:,} rows × {len(df.columns)} columns")
    
    
    if sample_size is None:
        print(f"  Using FULL dataset (all {full_size:,} rows)")
        return df, full_size
    
    # #validate sample_size
    # if sample_size <= 0:
    #     raise ValueError("sample_size must be greater than 0 or None for full dataset")
    
    # if sample_size >= full_size:
    #     print(f"  Sample size ({sample_size:,}) >= full size, using full dataset")
    #     return df, full_size
    
    # Sample the dataset
    df = df.sample(n=sample_size, random_state=random_state)
    print(f"  Sampled: {sample_size:,} rows (random_state={random_state})")
    
    return df, full_size


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Execute the full integrated pipeline with timing"""

    pipeline_start = time.time()
    
    # Sanitize dataset name for file paths
    dataset_filename = DATASET.replace(' ', '_').lower()
    

    
    #display configuration
    print("\n" + "═" * 80)
    print("  ⚙  PIPELINE CONFIGURATION")
    print("═" * 80)
    sample_text = f"{SAMPLE_SIZE:,}" if SAMPLE_SIZE else "FULL"
    print(f"  Dataset: {DATASET} | Sample: {sample_text} rows | Random State: {RANDOM_STATE} | Fuzzy Threshold: 0.80")
    
    #timing storage
    timings = {}
    
    # LOAD DATASET
    load_start = time.time()
    df, full_size = load_dataset(DATASET, SAMPLE_SIZE, RANDOM_STATE)
    timings['data_loading'] = time.time() - load_start
    print_timing("Data Loading Time", timings['data_loading'])
    


    # MODULE 1: DATA PROFILING 
    print_pipeline_header("📊 MODULE 1: DATA PROFILING")
    
    profiler = DataProfiler()
    
    start_time = time.time()
    profile = profiler.generate_profile(df)
    timings['data_profiling'] = time.time() - start_time
    
    profiler.print_summary()
    print_timing("Module 1 Time", timings['data_profiling'])
    


    # MODULE 2: FORMAT CORRECTION
    print_pipeline_header("🔧 MODULE 2: FORMAT CORRECTION")
    
    corrector = FormatCorrector()
    
    config = {
        'normalize_strings': True,
        'standardize_dates': True,
        'correct_types': True,
        'string_case': 'lower'
    }
    
    # Save original dataframe for before/after comparison
    df_before = df.copy()
    
    start_time = time.time()
    df_corrected = corrector.correct_formats(df, config)
    timings['format_correction'] = time.time() - start_time
    
    corrector.print_summary(df_before, df_corrected, num_rows=3)
    
    print_timing("Module 2 Time", timings['format_correction'])
    


    # MODULE 3: DUPLICATE DETECTION 
    print_pipeline_header("🔍 MODULE 3: DUPLICATE DETECTION")
    
    detector = DuplicateDetector()  # Uses default threshold of 0.80
    
    # Exact duplicate detection
    start_time = time.time()
    exact_duplicates = detector.detect_duplicates(df_corrected)
    timings['exact_duplicate_detection'] = time.time() - start_time
    
    exact_removed = 0
    if len(exact_duplicates) > 0:
        df_corrected = detector.remove_duplicates(df_corrected)
        exact_removed = detector.duplicates_removed
    
    # Fuzzy duplicate detection
    comparisons = len(df_corrected) * (len(df_corrected) - 1) // 2
    print(f"\n  Starting fuzzy detection (threshold: 0.80)...")
    print(f"  Comparisons needed: {comparisons:,} ")
    
    start_time = time.time()
    fuzzy_pairs = detector.detect_fuzzy_duplicates(df_corrected)
    timings['fuzzy_duplicate_detection'] = time.time() - start_time
    
    timings['duplicate_detection_total'] = timings['exact_duplicate_detection'] + timings['fuzzy_duplicate_detection']
    
    # Print summary
    detector.print_summary(dataset_name=DATASET)
    
    # Print timing
    print(f"\n  ⏱  Exact: {format_time(timings['exact_duplicate_detection'])} | Fuzzy: {format_time(timings['fuzzy_duplicate_detection'])} | Total: {format_time(timings['duplicate_detection_total'])}")
    
    #interactive fuzzy duplicate review (if any found)
    # comes after the summary and timing so user sees full context first
    if len(detector.fuzzy_duplicate_pairs) > 0:
        df_corrected = detector.remove_fuzzy_duplicates(df_corrected, interactive=True)
    

    # PIPELINE COMPLETE
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start
    timings['total_pipeline'] = total_time

    # EXECUTION TIME SUMMARY
    print("\n" + "═" * 80)
    print("  ⏱  EXECUTION TIME BREAKDOWN")
    print("═" * 80)
    
    modules = [
        ('Data Loading', timings['data_loading']),
        ('Module 1: Data Profiling', timings['data_profiling']),
        ('Module 2: Format Correction', timings['format_correction']),
        ('Module 3: Exact Detection', timings['exact_duplicate_detection']),
        ('Module 3: Fuzzy Detection', timings['fuzzy_duplicate_detection']),
    ]
    
    for module_name, module_time in modules:
        pct = (module_time / total_time * 100) if total_time > 0 else 0
        print(f"  {module_name:<35} {format_time(module_time):>18} ({pct:>5.1f}%)")
    
    print("  " + "─" * 76)
    print(f"  {'TOTAL PIPELINE TIME':<35} {format_time(total_time):>18} ")

    #SAVE CLEANED DATA
    cleaned_data_path = f'data/output/{dataset_filename}_pipeline_cleaned.csv'
    df_corrected.to_csv(cleaned_data_path, index=False)
    print(f"  ✓ Cleaned dataset saved to {cleaned_data_path}")
    
    #SAVE LOGS
    print("\n" + "═" * 80)
    print("  💾 SAVING LOGS")
    print("═" * 80 + "\n")
    
    #save individual module logs
    profiler.save_report(f'data/output/{dataset_filename}_pipeline_profile.json')
    corrector.save_corrections_log(f'data/output/{dataset_filename}_pipeline_format_corrections.json')
    detector.save_duplicates_log(f'data/output/{dataset_filename}_pipeline_duplicate_detection.json')
    
    #save compiled pipeline log
    import json
    pipeline_log = {
        'dataset': DATASET,
        'sample_size': SAMPLE_SIZE if SAMPLE_SIZE else 'FULL',
        'total_rows': len(df),
        'random_state': RANDOM_STATE,
        'total_execution_time_seconds': total_time,
        'timings': {
            'data_loading': timings['data_loading'],
            'data_profiling': timings['data_profiling'],
            'format_correction': timings['format_correction'],
            'exact_duplicate_detection': timings['exact_duplicate_detection'],
            'fuzzy_duplicate_detection': timings['fuzzy_duplicate_detection'],
            'total_pipeline': timings['total_pipeline']
        },
        'modules': {
            'profiling': {
                'profile_saved': f'{dataset_filename}_pipeline_profile.json'
            },
            'format_correction': {
                'log_saved': f'{dataset_filename}_pipeline_format_corrections.json'
            },
            'duplicate_detection': {
                'log_saved': f'{dataset_filename}_pipeline_duplicate_detection.json'
            }
        }
    }
    
    pipeline_log_path = f'data/output/{dataset_filename}_pipeline_summary.json'
    with open(pipeline_log_path, 'w') as f:
        json.dump(pipeline_log, f, indent=2)
    
    print(f"✓ Pipeline summary saved to {pipeline_log_path}")
    
    print("\n" + "═" * 80)
    print("  ✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("═" * 80 + "\n")
    
    return {
        'timings': timings,
        'profile': profile,
        'df_original': df,
        'df_corrected': df_corrected,
        'profiler': profiler,
        'corrector': corrector,
        'detector': detector
    }

# ENTRY POINT

if __name__ == "__main__":
    results = run_full_pipeline()
