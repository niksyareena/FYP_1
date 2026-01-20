"""
Inject duplicates in Adult dataset
"""

import pandas as pd
import numpy as np

#load format corrected data (following the order of the pipeline)
def load_data():
    print("Loading format-corrected Adult dataset...")
    df = pd.read_csv('data/output/adult_corrected.csv')
    print(f"✓ Loaded pre-corrected dataset: {len(df):,} rows")
    return df

def inject_duplicates(df, n_base=500, n_exact=5, n_fuzzy=100):
    """
    Create a controlled test dataset with known duplicates
    
    Args:
        df: Original Adult dataset
        n_base: Number of base rows to use (default: 500)
        n_exact: Number of exact duplicates to inject (default: 5, minimal for integration check)
        n_fuzzy: Number of fuzzy duplicates to inject (default: 100, main testing focus)
    
    Returns:
        Test dataframe, original indices mapping, and ground truth
    """
    print(f"\n{'='*70}")
    print("PRE-CLEANING DATASET")
    print(f"{'='*70}\n")
    
    #remove exact duplicates
    df_clean = df.drop_duplicates().copy()
    print(f"✓ Removed exact duplicates: {len(df)} → {len(df_clean)} rows")
    
    #sample base rows from clean dataset
    base_sample = df_clean.sample(n=n_base, random_state=42).reset_index(drop=True)
    print(f"✓ Sampled {n_base} clean base rows")
    
    print(f"\n{'='*70}")
    print("CREATING CONTROLLED TEST DATASET")
    print(f"{'='*70}")
    
    #track what we inject
    exact_duplicate_pairs = []
    fuzzy_duplicate_pairs = []
    
    #create exact duplicates
    print(f"\n{'='*70}")
    print("INJECTING EXACT DUPLICATES")
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
    print("INJECTING FUZZY DUPLICATES")
    print(f"{'='*70}\n")
    
    #pure typos only - no format issues (those are already handled by format correction module)
    #4 fields with 2-3 typo variations each for better coverage
    #note: format correction preserves hyphens, only converts to lowercase
    workclass_typos = {
        'private': ['privete', 'prvate'],
        'self-emp-not-inc': ['self-enp-not-inc', 'self-emp-npt-inc'],
        'self-emp-inc': ['self-enp-inc', 'self-emp-imc'],
        'federal-gov': ['federai-gov', 'fedreal-gov'],
        'state-gov': ['stare-gov', 'state-giv'],
        'local-gov': ['locsl-gov', 'local-giv']
    }
    
    education_typos = {
        'hs-grad': ['hs-gard', 'hs-grsd'],
        'bachelors': ['bachelers', 'batchelors'],
        'some-college': ['some-colege', 'sone-college'],
        'masters': ['mastres', 'masrers'],
        'assoc-voc': ['assoc-vic', 'assic-voc'],
        'assoc-acdm': ['assoc-acdn', 'assic-acdm'],
        '11th': ['11rh', '11ht'],
        '9th': ['9rh', '9ht']
    }
    
    occupation_typos = {
        'machine-op-inspct': ['machine-op-inspvt', 'machime-op-inspct'],
        'exec-managerial': ['exec-manageral', 'exec-manegerial'],
        'prof-specialty': ['prof-speciality', 'prof-specialy'],
        'craft-repair': ['craft-repiar', 'carft-repair'],
        'adm-clerical': ['adm-clercial', 'adn-clerical'],
        'sales': ['seles', 'saels'],
        'transport-moving': ['transport-movinr', 'transprot-moving'],
        'handlers-cleaners': ['handlers-clenaers', 'handlerz-cleaners'],
        'farming-fishing': ['farming-fishimg', 'farmimg-fishing'],
        'other-service': ['other-servise', 'othr-service'],
        'tech-support': ['tech-suport', 'teck-support']
    }
    
    marital_status_typos = {
        'married-civ-spouse': ['married-civ-spause', 'married-civ-spouce'],
        'never-married': ['never-maried', 'never-marrird'],
        'divorced': ['divorsed', 'divorved'],
        'separated': ['seperated', 'separeted'],
        'widowed': ['widowwd', 'widowd'],
        'married-spouse-absent': ['married-spouse-absemt', 'married-spouce-absent']
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
    print("Saving test dataset to 'data/output/test_duplicates_injected_adult.csv'...")
    final_df.to_csv('data/output/test_duplicates_injected_adult.csv', index=False)
    print("✓ Saved\n")
    
    return final_df, ground_truth

#run injection 
def main():
    print("Loading Adult dataset...")
    df = load_data()
    print(f"✓ Loaded {len(df):,} rows\n")
    
    #create test dataset with injected duplicates
    #only 5 exact duplicates for integration check (pandas duplicated() is already well-tested)
    #focus on 100 fuzzy duplicates (Levenshtein-based matching needs validation)
    df_test, ground_truth = inject_duplicates(
        df, 
        n_base=500,
        n_exact=5,
        n_fuzzy=100
    )

if __name__ == "__main__":
    main()

