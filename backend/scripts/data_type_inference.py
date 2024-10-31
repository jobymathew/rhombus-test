import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Union, Optional
import logging
from pathlib import Path
import dask.dataframe as dd
import warnings

class DataTypeInference:
    """A class to handle intelligent data type inference for pandas DataFrames."""
    
    def __init__(self, 
                 categorical_threshold: float = 0.5,
                 date_sample_size: int = 1000,
                 memory_efficient: bool = True,
                 min_memory_size: int = 1024 * 1024 * 500  # 500MB
                ):
        """
        Initialize the DataTypeInference class.
        
        Args:
            categorical_threshold: Ratio threshold for categorical conversion
            date_sample_size: Number of samples to check for date parsing
            memory_efficient: Whether to use memory efficient processing
            min_memory_size: Minimum file size in bytes to trigger memory efficient processing
        """
        self.categorical_threshold = categorical_threshold
        self.date_sample_size = date_sample_size
        self.memory_efficient = memory_efficient
        self.min_memory_size = min_memory_size
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Common date patterns
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # yyyy-mm-dd
            r'\d{2}/\d{2}/\d{4}',  # mm/dd/yyyy
            r'\d{2}-\d{2}-\d{4}',  # dd-mm-yyyy
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # m/d/yy
            r'\d{4}\d{2}\d{2}'     # yyyymmdd
        ]

    def _estimate_file_size(self, file_path: Union[str, Path]) -> int:
        """Estimate file size in bytes."""
        return Path(file_path).stat().st_size

    def _get_optimal_numeric_type(self, series: pd.Series) -> str:
        """
        Determine the most memory-efficient numeric type for a series.
        
        Args:
            series: pandas Series to analyze
            
        Returns:
            str: Optimal data type
        """
        
        # Check if the series has NaN values
        has_na = series.isna().any()
        
        

        # Get min and max values
        min_val = series.min()
        max_val = series.max()

        # If integers only and no NaNs, use integer types
        if series.dtype.kind in ['i', 'f']:
            if not has_na:
                # Check if the series only contains integer values
                if series.apply(lambda x: x.is_integer()).all():
                    # Assign optimal integer type
                    if min_val >= -128 and max_val <= 127:
                        return 'int8'
                    elif min_val >= -32768 and max_val <= 32767:
                        return 'int16'
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        return 'int32'
                    else:
                        return 'int64'
            else:
                # Handle nullable integers when NaNs are present
                try:
                    series_int64 = series.astype('Int64')  # Nullable integer type
                    return 'Int64'
                except ValueError:
                    pass  # Fall back if the conversion to Int64 fails
        
        # Check for float32 sufficiency if it's a float column
        if series.dtype.kind == 'f':
            float32_series = series.astype('float32')
            if (series - float32_series).abs().max() < 1e-6:
                return 'float32'

        return 'float64'
        
        # if series.dtype.kind not in ['i', 'f']:
        #     return series.dtype
            
        # # Get min and max values
        # min_val = series.min()
        # max_val = series.max()
        
        # # Check if integers only
        # if series.dtype.kind == 'f':
        #     if series.dropna().apply(lambda x: x.is_integer()).all():
        #         # Convert to integer type
        #         if min_val >= -128 and max_val <= 127:
        #             return 'int8'
        #         elif min_val >= -32768 and max_val <= 32767:
        #             return 'int16'
        #         elif min_val >= -2147483648 and max_val <= 2147483647:
        #             return 'int32'
        #         else:
        #             return 'int64'
        
        # # For floats, check if float32 is sufficient
        # if series.dtype.kind == 'f':
        #     float32_series = series.astype('float32')
        #     if (series - float32_series).abs().max() < 1e-6:
        #         return 'float32'
        
        # return 'float64'

    def _check_date_pattern(self, value: str) -> bool:
        """Check if a string matches any common date pattern."""
        return any(re.match(pattern, str(value)) for pattern in self.date_patterns)

    def _is_datetime_column(self, series: pd.Series) -> bool:
        """
        Check if a series contains datetime values.
        
        Args:
            series: pandas Series to analyze
            
        Returns:
            bool: True if series contains datetime values
        """
        if series.dtype == 'datetime64[ns]':
            return True
            
        if series.dtype != 'object':
            return False
            
        # Sample the series for efficiency
        sample = series.dropna().sample(
            n=min(self.date_sample_size, len(series)),
            random_state=42
        )
        
        # Check if sample matches date patterns
        matches = sample.astype(str).apply(self._check_date_pattern)
        return matches.mean() > 0.9  # 90% threshold

    def _infer_column_type(self, series: pd.Series, column_name: str) -> tuple:
        """
        Infer the appropriate data type for a single column.
        
        Args:
            series: pandas Series to analyze
            column_name: Name of the column
            
        Returns:
            tuple: (inferred_type, conversion_function)
        """
        # Convert common placeholders to NaN (e.g., "Not Available")
        # series = series.replace(["Not Available", "NA", "N/A", "null"], 2.0)
        
        
        # Check for datetime first
        if self._is_datetime_column(series):
            return 'datetime64[ns]', pd.to_datetime
        
        # Check for alphanumeric
        # If the series contains any non-numeric values, treat as string
        if series.astype(str).str.contains(r'[^\d.-]').any():
            print('True this one has alphanumeric')
        
        # Check if the series predominantly contains text
        # if series.dtype == 'object' and series.apply(lambda x: isinstance(x, str)).mean() > 0.9:
        #     # Convert to 'string' type if mostly strings
        #     return pd.StringDtype(), lambda x: x.astype(pd.StringDtype())
    
    # Try numeric conversion only if it does not look like text
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.notna().any():
            # Check if integer-like and no NaNs, suitable for ID columns like invoiceNo
            if numeric_series.dropna().apply(float.is_integer).all():
                optimal_type = self._get_optimal_numeric_type(numeric_series)
                return optimal_type, lambda x: pd.to_numeric(x, errors='coerce').astype(optimal_type)
            else:
                # Use float type when decimal values are present
                return 'float64', lambda x: pd.to_numeric(x, errors='coerce').astype('float64')    
        # Try numeric conversion
        # numeric_series = pd.to_numeric(series, errors='coerce')
        # if numeric_series.notna().any():
        #     optimal_type = self._get_optimal_numeric_type(numeric_series)
        #     return optimal_type, lambda x: pd.to_numeric(x, errors='coerce').astype(optimal_type)
            
        # Check for categorical
        if series.dtype == 'object':
            unique_ratio = len(series.unique()) / len(series)
            if unique_ratio < self.categorical_threshold:
                return 'category', pd.Categorical
            
            # Check if the series contains only strings and convert to string dtype
            # elif series.apply(lambda x: isinstance(x, str)).all():
                # return pd.StringDtype(), lambda x: x.astype(pd.StringDtype())
            
        # Convert to 'string' type if mostly strings and not categorical
        if series.apply(lambda x: isinstance(x, str)).mean() > 0.9:
            return pd.StringDtype(), lambda x: x.astype(pd.StringDtype())
            
            
                
        # Default to object type
        return 'object', str

    def process_file(self, 
                    file_path: Union[str, Path], 
                    chunk_size: Optional[int] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Process a CSV or Excel file and infer appropriate data types.
        
        Args:
            file_path: Path to the input file
            chunk_size: Size of chunks for processing large files
            **kwargs: Additional arguments passed to pd.read_csv or pd.read_excel
            
        Returns:
            pd.DataFrame: Processed DataFrame with inferred types
        """
        file_path = Path(file_path)
        file_size = self._estimate_file_size(file_path)
        
        self.logger.info(f"Processing file: {file_path}")
        self.logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Determine if we need memory-efficient processing
        use_chunks = self.memory_efficient and file_size > self.min_memory_size
        
        if use_chunks:
            return self._process_large_file(file_path, chunk_size, **kwargs)
        else:
            return self._process_small_file(file_path, **kwargs)

    def _process_small_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Process a small file that fits in memory."""
        # Read the file
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        return self.infer_and_convert_types(df)

    def _process_large_file(self, 
                          file_path: Path, 
                          chunk_size: Optional[int],
                          **kwargs) -> pd.DataFrame:
        """Process a large file using Dask for memory efficiency."""
        self.logger.info("Using Dask for memory-efficient processing")
        
        # Create Dask DataFrame
        ddf = dd.read_csv(file_path, **kwargs)
        
        # Compute column types using a sample
        sample_df = ddf.head(n=10000)
        inferred_types = {}
        
        for column in sample_df.columns:
            dtype, _ = self._infer_column_type(sample_df[column], column)
            inferred_types[column] = dtype
            
        # Apply types to full dataset
        ddf = ddf.astype(inferred_types)
        
        # Convert back to pandas
        return ddf.compute()

    def infer_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer and convert data types for all columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame with inferred types
        """
        result_df = df.copy()
        
        for column in df.columns:
            try:
                dtype, conversion_func = self._infer_column_type(df[column], column)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result_df[column] = conversion_func(df[column])
                    
                self.logger.info(f"Column '{column}' converted to {dtype}")
                
            except Exception as e:
                self.logger.warning(f"Error converting column '{column}': {str(e)}")
                continue
                
        return result_df

# Example usage
if __name__ == "__main__":
    # Initialize the converter
    dtype_converter = DataTypeInference(
        categorical_threshold=0.5,
        date_sample_size=1000,
        memory_efficient=True
    )
    
    # Process the file
    df = dtype_converter.process_file(
        'online_retail.csv',
        chunk_size=10000  # Optional: for large files
    )
    
    # Display results
    print("\nOriginal Data Types:")
    print(df.dtypes)
    
    # Memory usage information
    print("\nMemory Usage Per Column:")
    print(df.memory_usage(deep=True))