"""
Data loader module for insurance analytics project.
Handles loading and initial processing of pipe-delimited text files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class InsuranceDataLoader:
    """Loader for insurance data in pipe-delimited format."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_path : str, optional
            Path to the data file. If None, looks in default location.
        """
        self.data_path = data_path or 'data/raw/insurance_data.txt'
        self.data = None
        self.raw_columns = None
        
    def load_data(self, sample_size: int = None) -> pd.DataFrame:
        """
        Load the pipe-delimited insurance data.
        
        Parameters
        ----------
        sample_size : int, optional
            If provided, load only a sample of the data for testing.
            
        Returns
        -------
        pd.DataFrame
            Loaded insurance data.
        """
        print(f"📁 Loading data from: {self.data_path}")
        
        try:
            # Read the pipe-delimited file
            if sample_size:
                # For testing, read sample
                self.data = pd.read_csv(
                    self.data_path, 
                    sep='|',
                    nrows=sample_size,
                    low_memory=False
                )
                print(f" Loaded sample of {sample_size} rows")
            else:
                # Read full dataset
                self.data = pd.read_csv(
                    self.data_path, 
                    sep='|',
                    low_memory=False
                )
                print(f" Loaded full dataset: {self.data.shape[0]:,} rows, {self.data.shape[1]:,} columns")
            
            # Store original columns
            self.raw_columns = self.data.columns.tolist()
            
            # Basic info
            self._print_basic_info()
            
            return self.data
            
        except FileNotFoundError:
            print(f" File not found: {self.data_path}")
            print("Please ensure the data file exists in the specified location.")
            raise
        except Exception as e:
            print(f" Error loading data: {e}")
            raise
    
    def _print_basic_info(self):
        """Print basic information about the loaded data."""
        print("\n  BASIC DATA INFORMATION:")
        print(f"   Shape: {self.data.shape}")
        print(f"   Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"   Time period: {self.data['TransactionMonth'].min()} to {self.data['TransactionMonth'].max()}")
        
        # Check numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(f"   Numeric columns: {len(numeric_cols)}")
        
        # Check categorical columns
        cat_cols = self.data.select_dtypes(include=['object']).columns
        print(f"   Categorical columns: {len(cat_cols)}")
    
    def clean_column_names(self):
        """Clean column names (strip whitespace, lowercase, replace spaces)."""
        if self.data is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        print("\n Cleaning column names...")
        original_names = self.data.columns.tolist()
        
        # Clean column names
        self.data.columns = (
            self.data.columns
            .str.strip()  # Remove leading/trailing whitespace
            .str.lower()  # Convert to lowercase
            .str.replace(' ', '_')  # Replace spaces with underscores
            .str.replace('/', '_')  # Replace slashes
            .str.replace('(', '')  # Remove parentheses
            .str.replace(')', '')
        )
        
        print(f" Column names cleaned. Examples:")
        for orig, new in zip(original_names[:5], self.data.columns[:5]):
            print(f"   {orig} → {new}")
    
    def convert_data_types(self):
        """
        Convert columns to appropriate data types.
        - TransactionMonth to datetime
        - Numeric columns with special handling
        """
        if self.data is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        print("\n Converting data types...")
        
        # 1. Convert TransactionMonth to datetime
        if 'transactionmonth' in self.data.columns:
            self.data['transactionmonth'] = pd.to_datetime(
                self.data['transactionmonth'],
                errors='coerce'
            )
            print(f"    transactionmonth converted to datetime")
            
            # Check for NaT values
            nat_count = self.data['transactionmonth'].isna().sum()
            if nat_count > 0:
                print(f"     Found {nat_count} NaT values in transactionmonth")
        
        # 2. Identify numeric columns that might have string issues
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # 3. Handle TotalClaims which might have string representation
        if 'totalclaims' in self.data.columns:
            # Check if it's already numeric
            if self.data['totalclaims'].dtype == 'object':
                # Replace problematic values
                self.data['totalclaims'] = (
                    self.data['totalclaims']
                    .astype(str)
                    .str.replace('.000000000000', '0')  # Handle your specific format
                    .str.strip()
                    .replace('', '0')
                    .replace(' ', '0')
                )
                # Convert to numeric, coerce errors to NaN
                self.data['totalclaims'] = pd.to_numeric(
                    self.data['totalclaims'],
                    errors='coerce'
                )
                print(f"    totalclaims converted to numeric")
        
        # 4. Do the same for TotalPremium
        if 'totalpremium' in self.data.columns:
            if self.data['totalpremium'].dtype == 'object':
                self.data['totalpremium'] = (
                    self.data['totalpremium']
                    .astype(str)
                    .str.replace('.000000000000', '0')
                    .str.strip()
                    .replace('', '0')
                    .replace(' ', '0')
                )
                self.data['totalpremium'] = pd.to_numeric(
                    self.data['totalpremium'],
                    errors='coerce'
                )
                print(f"    totalpremium converted to numeric")
        
        print(f" Data type conversion complete")
    
    def get_missing_summary(self) -> pd.DataFrame:
        """
        Get summary of missing values.
        
        Returns
        -------
        pd.DataFrame
            Summary of missing values by column.
        """
        if self.data is None:
            print(" No data loaded. Call load_data() first.")
            return None
        
        missing_summary = pd.DataFrame({
            'column': self.data.columns,
            'missing_count': self.data.isnull().sum(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data)) * 100,
            'dtype': self.data.dtypes.values
        }).sort_values('missing_percentage', ascending=False)
        
        return missing_summary
    
    def get_numeric_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics for numeric columns.
        """
        if self.data is None:
            print(" No data loaded. Call load_data() first.")
            return None
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return self.data[numeric_cols].describe()
        else:
            print("  No numeric columns found")
            return pd.DataFrame()
    
    def save_processed_data(self, output_path: str = None):
        """
        Save processed data to parquet format for efficient storage.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the processed data. If None, uses default.
        """
        if self.data is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        if output_path is None:
            output_path = 'data/processed/insurance_data_processed.parquet'
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        self.data.to_parquet(output_path, index=False)
        print(f" Processed data saved to: {output_path}")
        print(f"   Size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")
    # Add this method to your InsuranceDataLoader class
    def clean_numeric_columns(self):
        """Clean numeric columns that have comma separators."""
        print("\n Cleaning numeric columns with comma separators...")
        
        # Columns that likely have comma issues based on your data preview
        comma_cols = ['mmcode', 'customvalueestimate', 'cubiccapacity', 
                    'kilowatts', 'suminsured']
        
        for col in comma_cols:
            if col in self.data.columns:
                # Check if column is string type and contains commas
                if self.data[col].dtype == 'object':
                    # Remove commas and convert to numeric
                    self.data[col] = (
                        self.data[col]
                        .astype(str)
                        .str.replace(',', '')
                        .str.replace(' ', '')
                    )
                    # Convert to numeric, coerce errors
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    print(f"   {col}: cleaned {self.data[col].isnull().sum()} problematic values")
        
        # Also handle other potential numeric columns
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                # Check if it looks like a number with commas
                sample = self.data[col].dropna().head(100) if not self.data[col].dropna().empty else []
                if any(isinstance(x, str) and ',' in str(x) for x in sample):
                    try:
                        self.data[col] = (
                            self.data[col]
                            .astype(str)
                            .str.replace(',', '')
                            .str.replace(' ', '')
                        )
                        self.data[col] = pd.to_numeric(self.data[col], errors='ignore')
                    except:
                        pass
        
        print(" Numeric column cleaning complete")


# Convenience function for quick loading
def load_insurance_data(data_path: str = None, sample_size: int = None, 
                       clean: bool = True, convert_dtypes: bool = True) -> pd.DataFrame:
    """
    Convenience function to load insurance data.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the data file.
    sample_size : int, optional
        Number of rows to load (for testing).
    clean : bool, default True
        Whether to clean column names.
    convert_dtypes : bool, default True
        Whether to convert data types.
        
    Returns
    -------
    pd.DataFrame
        Loaded insurance data.
    """
    loader = InsuranceDataLoader(data_path)
    data = loader.load_data(sample_size)
    
    if clean:
        loader.clean_column_names()
    
    if convert_dtypes:
        loader.convert_data_types()
    
    return loader.data


if __name__ == "__main__":
    # Test the loader
    print("🧪 Testing InsuranceDataLoader...")
    try:
        loader = InsuranceDataLoader()
        data = loader.load_data(sample_size=1000)  # Load sample for testing
        loader.clean_column_names()
        loader.convert_data_types()
        
        # Show missing values summary
        missing_df = loader.get_missing_summary()
        if missing_df is not None:
            print("\n Missing values summary (top 10):")
            print(missing_df.head(10))
        
        # Show numeric summary
        numeric_summary = loader.get_numeric_summary()
        if not numeric_summary.empty:
            print("\n Numeric columns summary:")
            print(numeric_summary.T)
        
        print("\n Data loader test completed successfully!")
        
    except Exception as e:
        print(f" Test failed: {e}")