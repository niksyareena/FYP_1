"""
Test duplicate detection on real Adult dataset
Run on sample first, then optionally on full dataset
"""

import pandas as pd
import time
from duplicate_detector import DuplicateDetector

def load_adult_data(sample_size, random_state, include_duplicates=True):
    """Load the format-corrected Adult dataset with option to include all duplicates"""
    print(f"\n{'='*70}")
    print("LOADING ADULT DATASET")
    print(f"{'='*70}\n")
    
    df = pd.read_csv('data/output/adult_corrected.csv')
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"✓ Using format-corrected data from pipeline")
    
    if include_duplicates:
        #find all duplicate rows in full dataset
        duplicate_mask = df.duplicated(keep=False)
        duplicate_indices = df[duplicate_mask].index.tolist()
        
        if len(duplicate_indices) > 0:
            print(f"✓ Found {len(duplicate_indices)} duplicate rows in full dataset")
            
            #get non-duplicate rows
            non_duplicate_indices = df[~duplicate_mask].index.tolist()
            
            #calculate remaining space for random rows
            remaining_size = max(0, sample_size - len(duplicate_indices))
            
            if remaining_size > 0:
                #sample random non-duplicate rows
                sampled_non_dup = pd.Series(non_duplicate_indices).sample(
                    n=min(remaining_size, len(non_duplicate_indices)), 
                    random_state=random_state
                ).tolist()
                
                #combine all duplicates + random non-duplicates
                selected_indices = duplicate_indices + sampled_non_dup
                df_sample = df.loc[selected_indices].sample(frac=1, random_state=random_state).reset_index(drop=True)
                
                print(f"✓ Sample: {len(duplicate_indices)} duplicates + {len(sampled_non_dup)} random rows = {len(df_sample):,} total")
            else:
                #sample size too small, only take duplicates
                df_sample = df.loc[duplicate_indices].sample(frac=1, random_state=random_state).reset_index(drop=True)
                print(f"✓ Sample: {len(df_sample):,} rows (all duplicates)")
        else:
            print(f"✓ No duplicates found in dataset")
            df_sample = df.sample(n=min(sample_size, len(df)), random_state=random_state)
            print(f"✓ Sample: {len(df_sample):,} random rows")
    else:
        #regular random sampling
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=random_state)
        print(f"✓ Sample: {len(df_sample):,} random rows")
    
    print(f"✓ Random state: {random_state}\n")
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
        #group duplicates together
        print(f"\n📋 Exact Duplicate Groups:")
        print(f"   Total duplicate groups found: {len(duplicate_rows)}")
        print(f"   Showing first 3 groups with FULL rows:\n")
        
        #find which rows each duplicate matches
        shown = 0
        seen_duplicates = set()
        
        for dup_idx in duplicate_rows.index:
            if dup_idx in seen_duplicates or shown >= 3:
                break
                
            #find all rows identical to this one
            dup_row = df.loc[dup_idx]
            matches = df[df.eq(dup_row).all(axis=1)].index.tolist()
            
            if len(matches) > 1:
                shown += 1
                seen_duplicates.update(matches)
                print(f"   {'='*65}")
                print(f"   GROUP {shown}: {len(matches)} identical rows")
                print(f"   Indices: {matches}")
                print(f"   {'='*65}")
                
                #show full rows
                for idx in matches:
                    print(f"\n   Row {idx} (full record):")
                    for col, val in df.loc[idx].items():
                        print(f"      {col}: {val}")
                print() 
    
    print(f"\n{'='*70}\n")
    return detector, duplicate_rows

def test_fuzzy_duplicates(df):
    """Test fuzzy duplicate detection (uses default threshold of 0.8)"""
    print(f"\n{'='*70}")
    print(f"STEP 2: FUZZY DUPLICATE DETECTION (threshold=0.80)")
    print(f"{'='*70}\n")
    
    detector = DuplicateDetector()  
    
    print(f"Detecting fuzzy duplicates...")
    print(f"Expected comparisons: {len(df) * (len(df) - 1) // 2:,}\n")
    
    start_time = time.time()
    fuzzy_pairs = detector.detect_fuzzy_duplicates(df)
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
            
            # Show full rows for comparison
            print(f"\n   Row {idx1}:")
            for col, val in df.iloc[idx1].items():
                print(f"      {col}: {val}")
            print(f"\n   Row {idx2}:")
            for col, val in df.iloc[idx2].items():
                print(f"      {col}: {val}")
            print()
    else:
        print(f"\n✓ No fuzzy duplicates found at threshold 0.80")
    
    print(f"\n{'='*70}\n")
    return detector, fuzzy_pairs

def main():
    print(f"\n{'='*70}")
    print("DUPLICATE DETECTION - ADULT DATASET")
    print(f"{'='*70}")
    
    overall_start = time.time()
    
    # Configuration 
    # Test different sample sizes to validate O(n²) complexity
    # Uncomment one sample size at a time to run each test
    
    #test 2
    # SAMPLE_SIZE = 1000
    # RANDOM_STATE = 101
    
    #test 3
    # SAMPLE_SIZE = 2000
    # RANDOM_STATE = 102
    
    #test 1
    SAMPLE_SIZE = 4000
    RANDOM_STATE = 42
    
    #load data
    df, total_rows = load_adult_data(sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    #detect exact duplicates
    exact_detector, exact_duplicates = test_exact_duplicates(df)
    
    #detect fuzzy duplicates
    fuzzy_detector, fuzzy_pairs = test_fuzzy_duplicates(df)
    
    #interactive fuzzy duplicate removal (if any found)
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
    
    #save results
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
    print(f"   Fuzzy duplicate pairs: {len(fuzzy_pairs):,} (threshold=0.80)")
    print(f"\n⏱️  Total Execution Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"\nExpected comparisons: {len(df) * (len(df) - 1) // 2:,}")
    print(f"Time per comparison: {(overall_time / (len(df) * (len(df) - 1) // 2)) * 1000:.4f} ms")
    

if __name__ == "__main__":
    main()
