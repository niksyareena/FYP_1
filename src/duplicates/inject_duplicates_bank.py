"""
Test duplicate detection with injected duplicates in Bank Marketing dataset
"""

import pandas as pd
import numpy as np

#load format corrected data (following the order of the pipeline)
def load_bank_data():
    df = pd.read_csv('data/output/bank_corrected.csv')
    print(f"✓ Loaded pre-corrected Bank Marketing dataset")
    return df

def create_test_dataset_with_injected_duplicates(df, n_base=500, n_exact=5, n_fuzzy=100):
    """
    Create a controlled test dataset with known duplicates
    
    Args:
        df: Pre-corrected Bank Marketing dataset
        n_base: Number of base rows to use
        n_exact: Number of exact duplicates to inject (integration check)
        n_fuzzy: Number of fuzzy duplicates to inject (main testing focus)
    
    Returns:
        Test dataframe and ground truth
    """
    print(f"\n{'='*70}")
    print("STEP 0: PRE-CLEANING DATASET")
    print(f"{'='*70}\n")
    
    #remove exact duplicates only
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
        source_idx = i
        duplicate_row = base_sample.iloc[source_idx].copy()
        exact_rows.append(duplicate_row)
        exact_duplicate_pairs.append((source_idx, len(base_sample) + i))
        
        print(f"Exact Duplicate #{i+1}:")
        print(f"  Source Index: {source_idx}")
        print(f"  Will be inserted at index: {len(base_sample) + i}")
        print(f"  Sample data: job={duplicate_row['job']}, education={duplicate_row['education']}, marital={duplicate_row['marital']}")
        print()
    
    #create fuzzy duplicates with variations
    print(f"\n{'='*70}")
    print("STEP 3: INJECTING FUZZY DUPLICATES")
    print(f"{'='*70}\n")
    
    #character-level typos only (no semantic variations or abbreviations)
    #2 variations per value for better distribution across fields
    job_typos = {
        'blue-collar': ['blu-collar', 'blue-colar'],
        'self-employed': ['self-emploied', 'self-employd'],
        'management': ['managment', 'mangement'],
        'housemaid': ['housemiad', 'housemaed'],
        'entrepreneur': ['entrepeneur', 'enterpreneur'],
        'admin.': ['admn.', 'adimn.'],
        'technician': ['technican', 'techncian'],
        'services': ['servies', 'sevices']
    }
    
    education_typos = {
        'university.degree': ['universty.degree', 'univeristy.degree'],
        'high.school': ['hgh.school', 'high.shool'],
        'professional.course': ['profesional.course', 'proffessional.course'],
        'basic.9y': ['basci.9y', 'basic.9 y'],
        'basic.6y': ['basci.6y', 'basic.6 y'],
        'basic.4y': ['basci.4y', 'basic.4 y']
    }
    
    marital_typos = {
        'married': ['marrird', 'maried'],
        'single': ['singel', 'sinlge'],
        'divorced': ['divorsed', 'divorved']
    }
    
    contact_typos = {
        'cellular': ['celular', 'celluar'],
        'telephone': ['telephon', 'telepone']
    }
    
    modified_rows = []
    for i in range(n_fuzzy):
        source_idx = n_exact + i
        clean_record = base_sample.iloc[source_idx].copy()
        messy_record = clean_record.copy()
        
        changes_made = []
        
        #apply job typos
        job_value = str(messy_record['job'])
        if job_value in job_typos:
            messy_record['job'] = str(np.random.choice(job_typos[job_value]))
            changes_made.append(f"job: '{clean_record['job']}' → '{messy_record['job']}'")
        
        #apply education typos
        edu_value = str(messy_record['education'])
        if edu_value in education_typos:
            messy_record['education'] = str(np.random.choice(education_typos[edu_value]))
            changes_made.append(f"education: '{clean_record['education']}' → '{messy_record['education']}'")
        
        #apply marital status typos
        marital_value = str(messy_record['marital'])
        if marital_value in marital_typos:
            messy_record['marital'] = str(np.random.choice(marital_typos[marital_value]))
            changes_made.append(f"marital: '{clean_record['marital']}' → '{messy_record['marital']}'")
        
        #apply contact typos
        contact_value = str(messy_record['contact'])
        if contact_value in contact_typos:
            messy_record['contact'] = str(np.random.choice(contact_typos[contact_value]))
            changes_made.append(f"contact: '{clean_record['contact']}' → '{messy_record['contact']}'")
        
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
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}\n")
    print(f"📊 Composition:")
    print(f"   Base rows: {n_base}")
    print(f"   Exact duplicates injected: {n_exact}")
    print(f"   Fuzzy duplicates injected: {n_fuzzy}")
    print(f"   Total dataset size: {len(final_df)}")
    print(f"\n📋 Ground Truth Pairs:")
    print(f"   Exact duplicate pairs: {len(exact_duplicate_pairs)}")
    print(f"   Fuzzy duplicate pairs: {len(fuzzy_duplicate_pairs)}")
    print(f"{'='*70}\n")
    
    # Save for inspection
    print("Saving test dataset to 'data/output/test_duplicates_injected_bank.csv'...")
    final_df.to_csv('data/output/test_duplicates_injected_bank.csv', index=False)
    print("✓ Saved\n")
    
    ground_truth = {
        'exact_pairs': exact_duplicate_pairs,
        'fuzzy_pairs': fuzzy_duplicate_pairs
    }
    
    return final_df, ground_truth

def main():
    """Run injection process"""
    print("Loading Bank Marketing dataset...")
    df = load_bank_data()
    print(f"✓ Loaded {len(df):,} rows\n")
    
    # Create test dataset with injected duplicates
    df_test, ground_truth = create_test_dataset_with_injected_duplicates(
        df, 
        n_base=500,
        n_exact=5,
        n_fuzzy=100
    )

if __name__ == "__main__":
    main()
