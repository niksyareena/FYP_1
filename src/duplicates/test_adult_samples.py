"""
Test duplicate detection on real Adult dataset
Run on sample first, then optionally on full dataset
"""

import pandas as pd
import time
from duplicate_detector import DuplicateDetector

def load_adult_data(sample_size, random_state):
    """Load the format-corrected Adult dataset"""
    print(f"\n{'='*70}")
    print("LOADING ADULT DATASET")
    print(f"{'='*70}\n")
    
    # Load the corrected dataset from format correction step
    df = pd.read_csv('data/output/adult_corrected.csv')
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"✓ Using format-corrected data from pipeline")
    
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    print(f"✓ Sampled {len(df_sample):,} rows (random_state={random_state})\n")
    return df_sample, len(df)

def test_exact_duplicates(df):
    """Test exact duplicate detection"""
    print(f"\n{'='*70}")
    print("STEP 1: EXACT DUPLICATE DETECTION")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector()
    
    print("Detecting exact duplicates...")
    start_time = time.time()
    duplicate_rows = detector.detect_duplicates(df)
    elapsed_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Exact duplicates found: {len(duplicate_rows):,}")
    print(f"   Duplicate percentage: {len(duplicate_rows)/len(df)*100:.2f}%")
    print(f"   Unique rows: {len(df) - len(duplicate_rows):,}")
    print(f"   Execution time: {elapsed_time:.2f} seconds")
    
    if len(duplicate_rows) > 0:
        # Group duplicates together
        print(f"\n📋 Exact Duplicate Groups:")
        print(f"   Total duplicate groups found: {len(duplicate_rows)}")
        print(f"   Showing first 3 groups with FULL rows:\n")
        
        # Find which rows each duplicate matches
        shown = 0
        seen_duplicates = set()
        
        for dup_idx in duplicate_rows.index:
            if dup_idx in seen_duplicates or shown >= 3:
                break
                
            # Find all rows identical to this one
            dup_row = df.loc[dup_idx]
            matches = df[df.eq(dup_row).all(axis=1)].index.tolist()
            
            if len(matches) > 1:
                shown += 1
                seen_duplicates.update(matches)
                print(f"   {'='*65}")
                print(f"   GROUP {shown}: {len(matches)} identical rows")
                print(f"   Indices: {matches}")
                print(f"   {'='*65}")
                
                # Show FULL rows for all duplicates in this group
                for idx in matches:
                    print(f"\n   Row {idx} (full record):")
                    for col, val in df.loc[idx].items():
                        print(f"      {col}: {val}")
                print()  # Extra space between groups
    
    print(f"\n{'='*70}\n")
    return detector, duplicate_rows

def test_fuzzy_duplicates(df, threshold=0.8):
    """Test fuzzy duplicate detection"""
    print(f"\n{'='*70}")
    print(f"STEP 2: FUZZY DUPLICATE DETECTION (threshold={threshold})")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector(fuzzy_threshold=threshold)
    
    print(f"Detecting fuzzy duplicates...")
    print(f"Note: This uses exhaustive comparison (all pairs)")
    print(f"Expected comparisons: {len(df) * (len(df) - 1) // 2:,}\n")
    
    start_time = time.time()
    fuzzy_pairs = detector.detect_fuzzy_duplicates(df, threshold=threshold)
    elapsed_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Total rows checked: {len(df):,}")
    print(f"   Fuzzy duplicate pairs found: {len(fuzzy_pairs):,}")
    affected_rows = len(set([idx for pair in fuzzy_pairs for idx in pair[:2]]))
    print(f"   Affected rows: {affected_rows:,}")
    print(f"   Execution time: {elapsed_time:.2f} seconds")
    
    if len(fuzzy_pairs) > 0:
        print(f"\n📋 Fuzzy Duplicate Pairs (showing top 10 by similarity):\n")
        sorted_pairs = sorted(fuzzy_pairs, key=lambda x: x[2], reverse=True)[:10]
        
        for pair_num, (idx1, idx2, sim) in enumerate(sorted_pairs, 1):
            print(f"   {'='*65}")
            print(f"   Pair {pair_num}: Row {idx1} ↔ Row {idx2} (Similarity: {sim:.2%})")
            print(f"   {'='*65}")
            
            # Show key columns for comparison
            key_cols = ['age', 'workclass', 'education', 'occupation', 'marital-status', 'race']
            available_cols = [col for col in key_cols if col in df.columns]
            
            print(f"\n   Row {idx1}:")
            for col in available_cols:
                print(f"      {col}: {df.iloc[idx1][col]}")
            print(f"\n   Row {idx2}:")
            for col in available_cols:
                print(f"      {col}: {df.iloc[idx2][col]}")
            print()
        
        print(f"\n⚠️  IMPORTANT: Review these pairs carefully!")
        print(f"   Fuzzy matching may detect both:")
        print(f"   - True duplicates (same person with typos)")
        print(f"   - Similar records (different people with similar attributes)")
    else:
        print(f"\n✓ No fuzzy duplicates found at threshold {threshold}")
    
    print(f"\n{'='*70}\n")
    return detector, fuzzy_pairs

def main():
    """Run duplicate detection on Adult dataset - Execution Time Analysis"""
    print(f"\n{'='*70}")
    print("DUPLICATE DETECTION - ADULT DATASET")
    print(f"{'='*70}")
    
    overall_start = time.time()
    
    # Configuration - Execution Time Analysis
    # Test different sample sizes to validate O(n²) complexity
    # Uncomment ONE sample size at a time to run each test
    
    THRESHOLD = 0.8
    
    #test 3
    SAMPLE_SIZE = 1000
    RANDOM_STATE = 101
    
    #test 2
    # SAMPLE_SIZE = 2000
    # RANDOM_STATE = 102
    
    #test 1
    # SAMPLE_SIZE = 4000
    # RANDOM_STATE = 42
    
    # Load data
    df, total_rows = load_adult_data(sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    # Step 1: Detect exact duplicates
    exact_detector, exact_duplicates = test_exact_duplicates(df)
    
    # Step 2: Detect fuzzy duplicates
    fuzzy_detector, fuzzy_pairs = test_fuzzy_duplicates(df, threshold=THRESHOLD)
    
    # Step 3: Interactive fuzzy duplicate removal (if any found)
    if len(fuzzy_pairs) > 0:
        print(f"\n{'='*70}")
        print("STEP 3: INTERACTIVE FUZZY DUPLICATE REMOVAL")
        print(f"{'='*70}\n")
        
        df_cleaned = fuzzy_detector.remove_fuzzy_duplicates(df, fuzzy_pairs=fuzzy_pairs, interactive=True)
        
        removed_count = len(df) - len(df_cleaned)
        print(f"\n✓ Rows removed: {removed_count}")
        print(f"✓ Rows remaining: {len(df_cleaned):,}")
    else:
        print(f"\n✓ No fuzzy duplicates found - no removal needed")
        df_cleaned = df
    
    # Save results using detector's built-in method
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    fuzzy_detector.save_duplicates_log('data/output/adult_duplicate_detection_log.json')
    
    overall_time = time.time() - overall_start
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nDataset: Adult (format-corrected)")
    print(f"Total rows in full dataset: {total_rows:,}")
    print(f"Rows tested: {len(df):,} (SAMPLE)")
    print(f"Random state: {RANDOM_STATE}")
    print(f"\nResults:")
    print(f"   Exact duplicates: {len(exact_duplicates):,}")
    print(f"   Fuzzy duplicate pairs: {len(fuzzy_pairs):,} (threshold={THRESHOLD})")
    print(f"\n⏱️  Total Execution Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"\nExpected comparisons: {len(df) * (len(df) - 1) // 2:,}")
    print(f"Time per comparison: {(overall_time / (len(df) * (len(df) - 1) // 2)) * 1000:.4f} ms")
    
    # Scaling reference
    print(f"\n{'='*70}")
    print("COMPLEXITY VALIDATION (O(n²) Scaling)")
    print(f"{'='*70}")
    print(f"Sample size doubling should result in ~4x execution time increase:")
    print(f"  1,000 rows → 2,000 rows: 4x increase expected")
    print(f"  2,000 rows → 4,000 rows: 4x increase expected")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
