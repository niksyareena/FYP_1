"""
Test duplicate detection with injected duplicates in Adult dataset
"""

import pandas as pd
import numpy as np
from duplicate_detector import DuplicateDetector

def load_adult_data():
    """load Adult dataset"""
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    df = pd.read_csv('datasets/adult/adult.data', 
                     names=column_names,
                     skipinitialspace=True)
    
    return df

def create_test_dataset_with_injected_duplicates(df, n_base=100, n_exact=10, n_fuzzy=15):
    """
    Create a controlled test dataset with known duplicates
    
    Args:
        df: Original Adult dataset
        n_base: Number of base rows to use
        n_exact: Number of exact duplicates to inject
        n_fuzzy: Number of fuzzy duplicates to inject
    
    Returns:
        Test dataframe, original indices mapping, and ground truth
    """
    print(f"\n{'='*70}")
    print("STEP 0: PRE-CLEANING BASE SAMPLE")
    print(f"{'='*70}\n")
    
    #get clean base dataset (remove exact duplicates)
    df_clean = df.drop_duplicates().copy()
    
    #sample more rows than needed to account for fuzzy duplicate removal
    oversample_factor = 3  #sample 3x to ensure we have enough after filtering
    candidate_sample = df_clean.sample(n=n_base * oversample_factor, random_state=42).reset_index(drop=True)
    print(f"✓ Sampled {len(candidate_sample)} candidate rows")
    
    #detect and remove fuzzy duplicates from candidate sample
    print(f"✓ Detecting pre-existing fuzzy duplicates in base sample...")
    detector = DuplicateDetector(fuzzy_threshold=0.75)
    fuzzy_pairs_in_base = detector.detect_fuzzy_duplicates(candidate_sample, threshold=0.75)
    
    #build set of indices to remove (keep first occurrence)
    indices_to_remove = set()
    for idx1, idx2, sim in fuzzy_pairs_in_base:
        indices_to_remove.add(max(idx1, idx2))  #remove later occurrence
    
    #filter to clean rows only
    clean_indices = [i for i in range(len(candidate_sample)) if i not in indices_to_remove]
    base_sample = candidate_sample.iloc[clean_indices[:n_base]].reset_index(drop=True)
    
    print(f"✓ Removed {len(indices_to_remove)} pre-existing fuzzy duplicates")
    print(f"✓ Final clean base sample: {len(base_sample)} rows")
    
    print(f"\n{'='*70}")
    print("STEP 1: CREATING CONTROLLED TEST DATASET")
    print(f"{'='*70}")
    
    print(f"\n✓ Base sample is now guaranteed free of fuzzy duplicates")
    
    #track what we inject
    exact_duplicate_pairs = []
    fuzzy_duplicate_pairs = []
    
    #create exact duplicates
    print(f"\n{'='*70}")
    print("STEP 2: INJECTING EXACT DUPLICATES")
    print(f"{'='*70}\n")
    
    exact_rows = []
    for i in range(n_exact):
        source_idx = i  #use first n_exact rows as sources
        duplicate_row = base_sample.iloc[source_idx].copy()
        exact_rows.append(duplicate_row)
        exact_duplicate_pairs.append((source_idx, len(base_sample) + i))
        
        print(f"Exact Duplicate #{i+1}:")
        print(f"  Source Index: {source_idx}")
        print(f"  Will be inserted at index: {len(base_sample) + i}")
        print(f"  Row data:")
        print(f"    {duplicate_row.to_dict()}")
        print()
    
    #create fuzzy duplicates with clear variations
    print(f"\n{'='*70}")
    print("STEP 3: INJECTING FUZZY DUPLICATES")
    print(f"{'='*70}\n")
    
    fuzzy_rows = []
    for i in range(n_fuzzy):
        source_idx = n_exact + i  #use next n_fuzzy rows as sources
        original_row = base_sample.iloc[source_idx].copy()
        fuzzy_row = original_row.copy()
        
        #apply specific variations to make it fuzzy
        variations_applied = []
        
        #variation 1: workclass (remove hyphens, change case, expand)
        if 'Private' in str(fuzzy_row['workclass']):
            fuzzy_row['workclass'] = 'private'
            variations_applied.append("workclass: 'Private' → 'private'")
        elif 'Self-emp-not-inc' in str(fuzzy_row['workclass']):
            fuzzy_row['workclass'] = 'Self emp not inc'
            variations_applied.append("workclass: 'Self-emp-not-inc' → 'Self emp not inc'")
        elif 'Federal-gov' in str(fuzzy_row['workclass']):
            fuzzy_row['workclass'] = 'federal gov'
            variations_applied.append("workclass: 'Federal-gov' → 'federal gov'")
        elif 'State-gov' in str(fuzzy_row['workclass']):
            fuzzy_row['workclass'] = 'state government'
            variations_applied.append("workclass: 'State-gov' → 'state government'")
        elif 'Local-gov' in str(fuzzy_row['workclass']):
            fuzzy_row['workclass'] = 'local gov'
            variations_applied.append("workclass: 'Local-gov' → 'local gov'")
        
        #variation 2: education (abbreviations, case changes, spelling)
        if 'HS-grad' in str(fuzzy_row['education']):
            new_val = str(np.random.choice(['hs grad', 'high school grad', 'HS graduate']))
            fuzzy_row['education'] = new_val
            variations_applied.append(f"education: 'HS-grad' → '{new_val}'")
        elif 'Bachelors' in str(fuzzy_row['education']):
            new_val = str(np.random.choice(['Bachelor', 'Bachelors degree', 'bachelors']))
            fuzzy_row['education'] = new_val
            variations_applied.append(f"education: 'Bachelors' → '{new_val}'")
        elif 'Some-college' in str(fuzzy_row['education']):
            fuzzy_row['education'] = 'Some college'
            variations_applied.append("education: 'Some-college' → 'Some college'")
        elif 'Masters' in str(fuzzy_row['education']):
            new_val = str(np.random.choice(['Master', 'Masters degree', 'masters']))
            fuzzy_row['education'] = new_val
            variations_applied.append(f"education: 'Masters' → '{new_val}'")
        elif 'Assoc-voc' in str(fuzzy_row['education']):
            fuzzy_row['education'] = 'Associate voc'
            variations_applied.append("education: 'Assoc-voc' → 'Associate voc'")
        
        #variation 3: occupation (remove hyphens, case changes)
        if 'Exec-managerial' in str(fuzzy_row['occupation']):
            fuzzy_row['occupation'] = 'Executive managerial'
            variations_applied.append("occupation: 'Exec-managerial' → 'Executive managerial'")
        elif 'Tech-support' in str(fuzzy_row['occupation']):
            fuzzy_row['occupation'] = 'technical support'
            variations_applied.append("occupation: 'Tech-support' → 'technical support'")
        elif 'Craft-repair' in str(fuzzy_row['occupation']):
            fuzzy_row['occupation'] = 'craft repair'
            variations_applied.append("occupation: 'Craft-repair' → 'craft repair'")
        elif 'Machine-op-inspct' in str(fuzzy_row['occupation']):
            fuzzy_row['occupation'] = 'Machine operator inspct'
            variations_applied.append("occupation: 'Machine-op-inspct' → 'Machine operator inspct'")
        elif 'Adm-clerical' in str(fuzzy_row['occupation']):
            fuzzy_row['occupation'] = 'Administrative clerical'
            variations_applied.append("occupation: 'Adm-clerical' → 'Administrative clerical'")
        elif 'Prof-specialty' in str(fuzzy_row['occupation']):
            fuzzy_row['occupation'] = 'Professional specialty'
            variations_applied.append("occupation: 'Prof-specialty' → 'Professional specialty'")
        
        #variation 4: marital-status (remove hyphens, expand abbreviations)
        if 'Never-married' in str(fuzzy_row['marital-status']):
            fuzzy_row['marital-status'] = 'never married'
            variations_applied.append("marital-status: 'Never-married' → 'never married'")
        elif 'Married-civ-spouse' in str(fuzzy_row['marital-status']):
            fuzzy_row['marital-status'] = 'Married civilian spouse'
            variations_applied.append("marital-status: 'Married-civ-spouse' → 'Married civilian spouse'")
        
        fuzzy_rows.append(fuzzy_row)
        fuzzy_duplicate_pairs.append((source_idx, len(base_sample) + n_exact + i))
        
        print(f"Fuzzy Duplicate #{i+1}:")
        print(f"  Source Index: {source_idx}")
        print(f"  Will be inserted at index: {len(base_sample) + n_exact + i}")
        print(f"  Original:")
        print(f"    {original_row.to_dict()}")
        print(f"  Modified:")
        print(f"    {fuzzy_row.to_dict()}")
        print(f"  Variations: {', '.join(variations_applied) if variations_applied else 'None (keeping same)'}")
        print()
    
    #combine all rows
    exact_df = pd.DataFrame(exact_rows)
    fuzzy_df = pd.DataFrame(fuzzy_rows)
    
    df_test = pd.concat([base_sample, exact_df, fuzzy_df], ignore_index=True)
    
    #ground truth
    ground_truth = {
        'base_size': n_base,
        'exact_duplicates': n_exact,
        'fuzzy_duplicates': n_fuzzy,
        'total_size': len(df_test),
        'exact_pairs': exact_duplicate_pairs,
        'fuzzy_pairs': fuzzy_duplicate_pairs
    }
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"\n📊 Composition:")
    print(f"   Base rows: {ground_truth['base_size']}")
    print(f"   Exact duplicates injected: {ground_truth['exact_duplicates']}")
    print(f"   Fuzzy duplicates injected: {ground_truth['fuzzy_duplicates']}")
    print(f"   Total dataset size: {ground_truth['total_size']}")
    
    print(f"\n📋 Ground Truth Pairs:")
    print(f"   Exact duplicate pairs: {len(exact_duplicate_pairs)}")
    print(f"   Fuzzy duplicate pairs: {len(fuzzy_duplicate_pairs)}")
    print(f"{'='*70}\n")
    
    return df_test, ground_truth

def verify_injected_duplicates(df, ground_truth):
    """verify that injected duplicates are actually in the dataset"""
    print(f"\n{'='*70}")
    print("STEP 4: VERIFICATION OF INJECTED DUPLICATES")
    print(f"{'='*70}\n")
    
    print("Verifying Exact Duplicates (All 10):")
    for i, (idx1, idx2) in enumerate(ground_truth['exact_pairs']):  
        row1 = df.iloc[idx1]
        row2 = df.iloc[idx2]
        is_identical = row1.equals(row2)
        print(f"  Pair {i+1}: Index {idx1} ↔ {idx2} - Identical: {is_identical}")
        print(f"    Row {idx1}:")
        print(f"      {row1.to_dict()}")
        print(f"    Row {idx2}:")
        print(f"      {row2.to_dict()}")
        print()
    
    print(f"\nVerifying Fuzzy Duplicates (All 15):")
    for i, (idx1, idx2) in enumerate(ground_truth['fuzzy_pairs']):  
        row1 = df.iloc[idx1]
        row2 = df.iloc[idx2]
        
        #show key fields
        print(f"  Pair {i+1}: Index {idx1} ↔ {idx2}")
        print(f"    Row {idx1}:")
        print(f"      {row1.to_dict()}")
        print(f"    Row {idx2}:")
        print(f"      {row2.to_dict()}")
        print()
    
    print(f"\n{'='*70}\n")
    
    return True

def main():
    """run controlled duplicate injection and show results"""
    
    #load data
    print("Loading Adult dataset...")
    df = load_adult_data()
    print(f"✓ Loaded {len(df):,} rows\n")
    
    #create test dataset with injected duplicates
    df_test, ground_truth = create_test_dataset_with_injected_duplicates(
        df, 
        n_base=100,      #use 100 base rows
        n_exact=10,      #inject 10 exact duplicates
        n_fuzzy=15       #inject 15 fuzzy duplicates
    )
    
    #verify injections
    verify_injected_duplicates(df_test, ground_truth)
    
    #save for inspection
    print("Saving test dataset to 'data/output/test_duplicates_injected.csv'...")
    df_test.to_csv('data/output/test_duplicates_injected.csv', index=False)
    print("✓ Saved\n")
 

if __name__ == "__main__":
    main()
