"""
Test duplicate detection with injected duplicates in Adult dataset
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from duplicates.duplicate_detector import DuplicateDetector
from format_correction.format_corrector import FormatCorrector

def load_adult_data():
    """Load Adult dataset and apply format correction"""
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    df = pd.read_csv('datasets/adult/adult.data', 
                     names=column_names,
                     skipinitialspace=True)
    
    #apply format correction to get clean, normalized data
    print("Applying format correction to original dataset...")
    corrector = FormatCorrector()
    df_corrected = corrector.normalize_strings(df, case='lower', normalize_punctuation=True)
    print(f"✓ Format correction applied (normalized punctuation and case)")
    
    return df_corrected

def create_test_dataset_with_injected_duplicates(df, n_base=500, n_exact=50, n_fuzzy=100):
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
    print("STEP 0: PRE-CLEANING DATASET")
    print(f"{'='*70}\n")
    
    #remove exact duplicates
    df_clean = df.drop_duplicates().copy()
    print(f"✓ Removed exact duplicates: {len(df)} → {len(df_clean)} rows")
    
    #sample base rows from clean dataset
    base_sample = df_clean.sample(n=n_base, random_state=42).reset_index(drop=True)
    print(f"✓ Sampled {n_base} clean base rows")
    
    print(f"\n{'='*70}")
    print("STEP 1: CREATING CONTROLLED TEST DATASET")
    print(f"{'='*70}")
    
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
    
    #create fuzzy duplicates with variations
    print(f"\n{'='*70}")
    print("STEP 3: INJECTING FUZZY DUPLICATES")
    print(f"{'='*70}\n")
    
    #pure typos only - no format issues (those are handled by format correction module)
    #4 fields with 2-3 typo variations each for better coverage
    workclass_typos = {
        'private': ['privete', 'prvate'],
        'self emp not inc': ['self enp not inc', 'self emp npt inc'],
        'federal gov': ['federai gov', 'fedreal gov'],
        'state gov': ['stare gov', 'state giv'],
        'local gov': ['locsl gov', 'local giv']
    }
    
    education_typos = {
        'hs grad': ['hs gard', 'hs grsd'],
        'bachelors': ['bachelers', 'batchelors'],
        'some college': ['some colege', 'sone college'],
        'masters': ['mastres', 'masrers'],
        'assoc voc': ['assoc vic', 'assic voc'],
        'assoc acdm': ['assoc acdn', 'assic acdm']
    }
    
    occupation_typos = {
        'machine op inspct': ['machine op inspvt', 'machime op inspct'],
        'exec managerial': ['exec manageral', 'exec manegerial'],
        'prof specialty': ['prof speciality', 'prof specialy'],
        'craft repair': ['craft repiar', 'carft repair'],
        'adm clerical': ['adm clerical', 'adm clercal'],
        'sales': ['seles', 'saels'],
        'transport moving': ['transport movinr', 'transprot moving'],
        'handlers cleaners': ['handlers clenaers', 'handlerz cleaners'],
        'farming fishing': ['farming fishimg', 'farmimg fishing']
    }
    
    marital_status_typos = {
        'married civ spouse': ['married civ spause', 'married civ spouce'],
        'never married': ['never maried', 'never marrird'],
        'divorced': ['divorsed', 'divorved'],
        'separated': ['seperated', 'separeted'],
        'widowed': ['widowwd', 'widowd']
    }
    
    modified_rows = []
    for i in range(n_fuzzy):
        source_idx = n_exact + i
        clean_record = base_sample.iloc[source_idx].copy()
        messy_record = clean_record.copy()
        
        changes_made = []
        
        #apply workclass typos
        wc_value = str(messy_record['workclass'])
        if wc_value in workclass_typos:
            messy_record['workclass'] = str(np.random.choice(workclass_typos[wc_value]))
            changes_made.append(f"workclass: '{clean_record['workclass']}' → '{messy_record['workclass']}'")
        
        #apply education typos
        edu_value = str(messy_record['education'])
        if edu_value in education_typos:
            messy_record['education'] = str(np.random.choice(education_typos[edu_value]))
            changes_made.append(f"education: '{clean_record['education']}' → '{messy_record['education']}'")
        
        #apply occupation typos
        occ_value = str(messy_record['occupation'])
        if occ_value in occupation_typos:
            messy_record['occupation'] = str(np.random.choice(occupation_typos[occ_value]))
            changes_made.append(f"occupation: '{clean_record['occupation']}' → '{messy_record['occupation']}'")
        
        #apply marital-status typos
        marital_value = str(messy_record['marital-status'])
        if marital_value in marital_status_typos:
            messy_record['marital-status'] = str(np.random.choice(marital_status_typos[marital_value]))
            changes_made.append(f"marital-status: '{clean_record['marital-status']}' → '{messy_record['marital-status']}'")
        
        modified_rows.append(messy_record)
        fuzzy_duplicate_pairs.append((source_idx, len(base_sample) + n_exact + i))
        
        print(f"Fuzzy Duplicate #{i+1}:")
        print(f"  Source Index: {source_idx}")
        print(f"  Will be inserted at index: {len(base_sample) + n_exact + i}")
        if changes_made:
            for change in changes_made:
                print(f"    {change}")
        else:
            print(f"    no variations applied (keeping original values)")
        print()
    
    #combine all rows
    exact_df = pd.DataFrame(exact_rows)
    modified_df = pd.DataFrame(modified_rows)
    
    final_df = pd.concat([base_sample, exact_df, modified_df], ignore_index=True)
    
    #ground truth
    ground_truth = {
        'base_size': n_base,
        'exact_duplicates': n_exact,
        'fuzzy_duplicates': n_fuzzy,
        'total_size': len(final_df),
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
    
    #save for inspection
    print("Saving test dataset to 'data/output/test_duplicates_injected.csv'...")
    final_df.to_csv('data/output/test_duplicates_injected.csv', index=False)
    print("✓ Saved\n")
    
    return final_df, ground_truth


def main():
    """Run injection process"""
    print("Loading Adult dataset...")
    df = load_adult_data()
    print(f"✓ Loaded {len(df):,} rows\n")
    
    #create test dataset with injected duplicates
    df_test, ground_truth = create_test_dataset_with_injected_duplicates(
        df, 
        n_base=500,
        n_exact=50,
        n_fuzzy=100
    )

if __name__ == "__main__":
    main()

