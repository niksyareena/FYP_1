"""
Format Corrector Module
Applies rule-based corrections to common data formatting issues
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import re


class FormatCorrector:
    """
    Rule-based format correction for:
    - String normalization (casing, whitespace)
    - Date standardization (ISO 8601 format)
    - Type correction (numeric, datetime)
    """
    
    def __init__(self):
        self.corrections_log = []
    
    def correct_formats(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply all format corrections to dataframe
        
        Args:
            df: Input dataframe
            config: Optional configuration dict with keys:
                - normalize_strings: bool (default True)
                - standardize_dates: bool (default True)
                - correct_types: bool (default True)
                - string_case: 'lower'|'upper'|'title' (default 'lower')
                - date_columns: List[str] (auto-detect if None)
                
        Returns:
            Corrected dataframe
        """
        if config is None:
            config = {}
        
        df_corrected = df.copy()
        self.corrections_log = []
        
        #string normalization
        if config.get('normalize_strings', True):
            df_corrected = self.normalize_strings(
                df_corrected, 
                case=config.get('string_case', 'lower')
            )
        
        #date standardization
        if config.get('standardize_dates', True):
            date_cols = config.get('date_columns', None)
            df_corrected = self.standardize_dates(df_corrected, date_cols)
        
        #type correction
        if config.get('correct_types', True):
            df_corrected = self.correct_types(df_corrected)
        
        return df_corrected
    
    def normalize_strings(self, df: pd.DataFrame, case: str = 'lower') -> pd.DataFrame:
        """
        Normalize string columns:
        - Trim whitespace
        - Normalize casing
        - Remove extra internal spaces
        """
        df_normalized = df.copy()
        string_cols = df.select_dtypes(include=['object']).columns
        
        for col in string_cols:
            original = df[col].copy()
            
            #skip if column has no string values
            if not df[col].apply(lambda x: isinstance(x, str)).any():
                continue
            
            #trim whitespace
            df_normalized[col] = df_normalized[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            
            #normalize internal whitespace (multiple spaces -> single space)
            df_normalized[col] = df_normalized[col].apply(
                lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x
            )
            
            #apply casing
            if case == 'lower':
                df_normalized[col] = df_normalized[col].apply(
                    lambda x: x.lower() if isinstance(x, str) else x
                )
            elif case == 'upper':
                df_normalized[col] = df_normalized[col].apply(
                    lambda x: x.upper() if isinstance(x, str) else x
                )
            elif case == 'title':
                df_normalized[col] = df_normalized[col].apply(
                    lambda x: x.title() if isinstance(x, str) else x
                )
            
            #log changes
            changes = (original != df_normalized[col]).sum()
            if changes > 0:
                self.corrections_log.append({
                    'column': col,
                    'operation': f'string_normalization (case={case})',
                    'rows_affected': int(changes)
                })
        
        return df_normalized
    
    #standardize date columns to dd/mm/yyyy format
    def standardize_dates(self, df: pd.DataFrame, date_columns: Optional[List[str]] = None) -> pd.DataFrame:
        
        df_standardized = df.copy()
        
        #auto-detect date columns if not provided
        if date_columns is None:
            date_columns = self.detect_date_columns(df)
        
        for col in date_columns:
            if col not in df.columns:
                continue
            
            original = df[col].copy()
            
            try:
                #try multiple date parsing strategies
                def parse_date(date_str):
                    if pd.isna(date_str):
                        return pd.NaT
                    
                    #try common date formats explicitly
                    formats = [
                        '%Y-%m-%d',      # 2020-01-15
                        '%d-%m-%Y',      # 15-01-2020
                        '%m/%d/%Y',      # 01/15/2020
                        '%d/%m/%Y',      # 15/01/2020
                        '%d-%b-%Y',      # 15-Jan-2020
                        '%d-%B-%Y',      # 15-January-2020
                        '%Y.%m.%d',      # 2020.01.15
                        '%d.%m.%Y',      # 15.01.2020
                    ]
                    
                    for fmt in formats:
                        try:
                            return pd.to_datetime(date_str, format=fmt)
                        except:
                            continue
                    
                    #if no format matches, try general parsing
                    try:
                        return pd.to_datetime(date_str)
                    except:
                        return pd.NaT
                
                #apply parsing to each value
                df_standardized[col] = df[col].apply(parse_date)
                
                #convert to dd/mm/yyyy format string (handle NaT values)
                df_standardized[col] = df_standardized[col].apply(
                    lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else x
                )
                
                #count successful conversions
                changes = (~original.isna() & (original.astype(str) != df_standardized[col])).sum()
                
                if changes > 0:
                    self.corrections_log.append({
                        'column': col,
                        'operation': 'date_standardization (dd/mm/yyyy)',
                        'rows_affected': int(changes)
                    })
            
            except Exception as e:
                #skip if conversion fails
                continue
        
        return df_standardized
    
    def detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect columns that might contain dates based on:
        - Column name patterns (date, time, year, etc.)
        - Sample values
        """
        date_cols = []
        
        #name-based detection
        date_keywords = ['date', 'time', 'year', 'month', 'day', 'dob', 'birth', 'created', 'updated']
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            #check column name
            if any(keyword in col_lower for keyword in date_keywords):
                date_cols.append(col)
                continue
            
            #check if column can be parsed as dates
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    try:
                        parsed = pd.to_datetime(sample, errors='coerce')
                        #if >50% of samples parse successfully, likely a date column
                        valid_dates = pd.notna(parsed).sum()
                        if valid_dates / len(sample) > 0.5:
                            date_cols.append(col)
                    except:
                        continue
        
        return date_cols
    
    def correct_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct data types for columns that are misclassified:
        - Numeric columns stored as strings
        - Boolean columns stored as strings
        """
        df_corrected = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                original = df[col].copy()
                
                #try numeric conversion
                try:
                    #remove common non-numeric characters
                    cleaned = df[col].apply(
                        lambda x: str(x).replace(',', '').replace('$', '').strip() 
                        if isinstance(x, str) else x
                    )
                    
                    numeric_converted = pd.to_numeric(cleaned, errors='coerce')
                    
                    #if >80% of non-null values convert successfully, apply conversion
                    non_null_count = df[col].notna().sum()
                    if non_null_count > 0:
                        success_rate = pd.notna(numeric_converted).sum() / non_null_count
                        
                        if success_rate > 0.8:
                            df_corrected[col] = numeric_converted
                            
                            changes = (~original.isna() & (original.astype(str) != df_corrected[col].astype(str))).sum()
                            if changes > 0:
                                self.corrections_log.append({
                                    'column': col,
                                    'operation': 'type_correction (object -> numeric)',
                                    'rows_affected': int(changes)
                                })
                            continue
                except:
                    pass
                
                #try boolean conversion
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2:
                    #check for boolean patterns
                    bool_patterns = {
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        '1': True, '0': False,
                        't': True, 'f': False,
                        'y': True, 'n': False
                    }
                    
                    #normalize and check
                    normalized_vals = {str(v).lower().strip() for v in unique_vals}
                    if normalized_vals.issubset(set(bool_patterns.keys())):
                        df_corrected[col] = df[col].apply(
                            lambda x: bool_patterns.get(str(x).lower().strip()) if pd.notna(x) else x
                        )
                        
                        self.corrections_log.append({
                            'column': col,
                            'operation': 'type_correction (object -> bool)',
                            'rows_affected': int(df[col].notna().sum())
                        })
        
        return df_corrected
    
    def get_corrections_summary(self) -> pd.DataFrame:
        
        if not self.corrections_log:
            return pd.DataFrame(columns=['column', 'operation', 'rows_affected'])
        
        return pd.DataFrame(self.corrections_log)
    
    def save_corrections_log(self, filepath: str):
       
        import json
        
        log_data = {
            'total_corrections': len(self.corrections_log),
            'total_rows_affected': sum(item['rows_affected'] for item in self.corrections_log) if self.corrections_log else 0,
            'corrections': self.corrections_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✓ Corrections log saved to {filepath}")
    
    def print_summary(self):
        
        if not self.corrections_log:
            print("No corrections were applied.")
            return
        
        print("=" * 70)
        print("FORMAT CORRECTIONS SUMMARY".center(70))
        print("=" * 70)
        
        summary_df = self.get_corrections_summary()
        
        print(f"\n📝 Total Corrections: {len(self.corrections_log)}")
        print(f"   Total Rows Affected: {summary_df['rows_affected'].sum():,}")
        
        print(f"\n🔧 CORRECTIONS BY COLUMN")
        print(f"   {'Column':<25} {'Operation':<30} {'Rows Affected':<15}")
        print(f"   {'-'*25} {'-'*30} {'-'*15}")
        
        for _, row in summary_df.iterrows():
            print(f"   {row['column']:<25} {row['operation']:<30} {row['rows_affected']:<15,}")
        
        print("\n" + "=" * 70)
