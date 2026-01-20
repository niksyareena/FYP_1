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
        
        #standardize missing values first (always run this)
        df_corrected = self.standardize_missing_values(df_corrected)
        
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
    
    def standardize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize all missing value representations to proper pandas NA/NaN
        
        Converts common missing value markers to pandas NA:
        - '?', 'NA', 'N/A', 'null', 'NULL', 'None', 'nan', 'NaN', ''
        - Whitespace-only strings
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with standardized missing values
        """
        df_standardized = df.copy()
        
        #define missing value markers (matching data profiler)
        missing_markers = ['?', 'NA', 'N/A', 'null', 'NULL', 'None', 'nan', 'NaN', '']
        
        total_replacements = 0
        
        for col in df.columns:
            if df[col].dtype == 'object':
                #count replacements before
                original = df[col].copy()
                
                #replace missing markers with pd.NA
                def standardize_value(x):
                    if pd.isna(x):
                        return pd.NA
                    if isinstance(x, str):
                        x_stripped = x.strip()
                        if x_stripped in missing_markers or x_stripped == '':
                            return pd.NA
                    return x
                
                df_standardized[col] = df_standardized[col].apply(standardize_value)
                
                #count changes
                changes = (~original.isna() & df_standardized[col].isna()).sum()
                if changes > 0:
                    total_replacements += changes
        
        if total_replacements > 0:
            self.corrections_log.append({
                'column': 'all_columns',
                'operation': 'missing_value_standardization',
                'cells_affected': int(total_replacements)
            })
        
        return df_standardized
    
    def is_numeric_range_column(self, series: pd.Series) -> bool:
        """
        Detect if a column contains numeric ranges (e.g., '30-39', '10-14')
        
        Args:
            series: pandas Series to check
            
        Returns:
            True if column appears to contain numeric ranges
        """
        #sample non-null values
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        
        #check if values match numeric range pattern (digit-digit)
        range_pattern = re.compile(r'^\d+[-]\d+$')
        matching_count = sum(1 for val in sample if isinstance(val, str) and range_pattern.match(val.strip()))
        
        #if >50% of samples are numeric ranges, consider it a range column
        return (matching_count / len(sample)) > 0.5
    
    def is_currency_or_numeric_notation_column(self, series: pd.Series) -> bool:
        """
        Detect if a column contains currency or numeric notation (e.g., '$50K', '<=50K', '>100M')
        These should preserve their casing and special characters.
        
        Args:
            series: pandas Series to check
            
        Returns:
            True if column appears to contain currency/numeric notation
        """
        #sample non-null values
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        
        #patterns for currency/numeric notation
        # - Currency symbols: $, €, £, ¥
        # - Unit suffixes: K, M, B (thousands, millions, billions)
        # - Comparison operators: <, >, <=, >=, =
        # - Percentages: %
        notation_pattern = re.compile(r'[$€£¥%]|[<>=]+|\d+[KMBkmb]\b')
        matching_count = sum(1 for val in sample if isinstance(val, str) and notation_pattern.search(val))
        
        #if >30% of samples contain notation, consider it a notation column
        return (matching_count / len(sample)) > 0.3
    
    def normalize_strings(self, df: pd.DataFrame, case: str = 'lower', 
                         normalize_punctuation: bool = True) -> pd.DataFrame:
        """
        Normalize string columns:
        - Trim whitespace
        - Normalize casing
        - Remove extra internal spaces
        - Optionally normalize punctuation/delimiters (dots, hyphens, underscores → spaces)
        
        Args:
            df: Input dataframe
            case: 'lower', 'upper', or 'title'
            normalize_punctuation: If True, converts punctuation to spaces for consistency
            
        Returns:
            Normalized dataframe
        """
        df_normalized = df.copy()
        string_cols = df.select_dtypes(include=['object']).columns
        
        for col in string_cols:
            #capture input state for this operation (not original df input)
            col_before = df_normalized[col].copy()
            
            #skip if column has no string values
            if not df[col].apply(lambda x: isinstance(x, str)).any():
                continue
            
            #check if this column contains numeric ranges or currency/numeric notation
            is_range_column = self.is_numeric_range_column(df[col])
            is_notation_column = self.is_currency_or_numeric_notation_column(df[col])
            
            #trim whitespace
            df_normalized[col] = df_normalized[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            
            #conservative punctuation normalization: only fix inconsistent delimiter usage
            #preserve hyphens by default (meaningful in ranges, compound words, etc)
            punctuation_normalized = False
            if normalize_punctuation and not is_range_column and not is_notation_column:
                #detect if column has mixed delimiter usage (both dots and underscores, etc.)
                sample = df_normalized[col].dropna().head(100)
                has_dots = any('.' in str(v) for v in sample if isinstance(v, str))
                has_underscores = any('_' in str(v) for v in sample if isinstance(v, str))
                has_multiple_delimiters = sum([has_dots, has_underscores]) > 1
                
                #only normalize if there's mixed delimiter usage indicating inconsistency
                if has_multiple_delimiters:
                    df_normalized[col] = df_normalized[col].apply(
                        lambda x: x.replace('.', ' ').replace('_', ' ') if isinstance(x, str) else x
                    )
                    punctuation_normalized = True
            
            #normalize internal whitespace (multiple spaces -> single space)
            df_normalized[col] = df_normalized[col].apply(
                lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x
            )
            
            #apply casing (skip for currency/numeric notation columns)
            if not is_notation_column:
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
            
            #log changes (exclude NA comparisons which count as not equal)
            mask = col_before.notna() & df_normalized[col].notna()
            changes = (col_before[mask] != df_normalized[col][mask]).sum()
            if changes > 0:
                #determine which operations actually made changes
                operations_applied = []
                
                #add case normalization if it was applied
                if not is_notation_column and case in ['lower', 'upper', 'title']:
                    operations_applied.append(f'case={case}')
                
                #add punctuation normalization if it was actually applied
                if punctuation_normalized:
                    operations_applied.append('punctuation=normalized')
                
                if operations_applied:
                    op_str = ', '.join(operations_applied)
                    self.corrections_log.append({
                        'column': col,
                        'operation': f'string_normalization ({op_str})',
                        'cells_affected': int(changes)
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
                def parse_date(date_str: Any) -> Any:
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
                df_standardized[col] = df_standardized[col].apply(parse_date)
                
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
                        'cells_affected': int(changes)
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
                        parsed = pd.to_datetime(sample, errors='coerce', format='mixed')
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
                                    'cells_affected': int(changes)
                                })
                            continue
                except:
                    pass
                
                #try boolean conversion
                #first, filter out common missing value markers
                missing_markers = ['?', 'NA', 'N/A', 'null', 'NULL', 'None', 'nan', 'NaN', '']
                unique_vals = [v for v in df[col].dropna().unique() if str(v).strip() not in missing_markers]
                
                if len(unique_vals) <= 2 and len(unique_vals) > 0:
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
                        #apply conversion, treating missing markers as NaN
                        def convert_to_bool(x):
                            if pd.notna(x):
                                x_str = str(x).strip()
                                if x_str in missing_markers:
                                    return pd.NA
                                return bool_patterns.get(x_str.lower())
                            return x
                        
                        df_corrected[col] = df_corrected[col].apply(convert_to_bool)
                        
                        self.corrections_log.append({
                            'column': col,
                            'operation': 'type_correction (object -> bool)',
                            'cells_affected': int(df[col].notna().sum())
                        })
        
        return df_corrected
    
    def get_corrections_summary(self) -> pd.DataFrame:
        
        if not self.corrections_log:
            return pd.DataFrame(columns=['column', 'operation', 'cells_affected'])
        
        return pd.DataFrame(self.corrections_log)
    
    def save_corrections_log(self, filepath: str):
       
        import json
        
        log_data = {
            'total_corrections': len(self.corrections_log),
            'total_cells_affected': sum(item['cells_affected'] for item in self.corrections_log) if self.corrections_log else 0,
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
        
        print(f"\n🔧 CORRECTIONS BY COLUMN")
        print(f"   {'Column':<25} {'Operation':<30}")
        print(f"   {'-'*25} {'-'*30}")
        
        for _, row in summary_df.iterrows():
            print(f"   {row['column']:<25} {row['operation']:<30}")
        
        print("\n" + "=" * 70)
