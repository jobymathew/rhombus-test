# infer_data_types.py
import pandas as pd

import pandas as pd
import numpy as np

def all_values_end_with_zero_or_nan(df, column_name):
  """
  Checks if all values in a specified column are either numeric strings ending with ".0" or NaN.

  Args:
    df: The Pandas DataFrame.
    column_name: The name of the column to check.

  Returns:
    True if all values meet the condition, False otherwise.
  """

  # Check if all values are either NaN or numeric strings ending with '.0'
  return df[column_name].isnull().all() or df[column_name].astype(str).str.endswith('.0').all()

# Example usage:
df = pd.DataFrame({'A': ['1.0', '2.0', np.nan, '3.0']})

if all_values_end_with_zero_or_nan(df, 'A'):
  print("All values in column A are either '.0' or NaN")
else:
  print("Not all values in column A are either '.0' or NaN")

def convert_to_int_float(df, column_name):
    """
    Converts a column to the appropriate numeric type (int64 or float64).

    Args:
        df: The Pandas DataFrame.
        column_name: The name of the column to process.

    Returns:
        The modified DataFrame.
    """

    df[column_name] = df[column_name].astype(str).str.rstrip('.0')
    print('Int to float conversion')
    print(df[column_name])


    # return df[column_name]

def is_integer_column(df, column_name):
    """
    Checks if a column in a DataFrame contains only integer values.

    Args:
        df: The DataFrame.
        column_name: The name of the column to check.

    Returns:
        True if the column contains only integer values, False otherwise.
    """

    # return np.allclose(df[column_name], df[column_name].round())
    return df[column_name].apply(float.is_integer).all()


def is_numeric_column(df, column_name):
    """
    Checks if a column in a DataFrame is numeric.

    Args:
        df: The DataFrame.
        column_name: The name of the column to check.

    Returns:
        True if the column is numeric, False otherwise.
    """

    try:
        new_col = pd.to_numeric(df[column_name], errors='coerce')
        # print(new_col.head())
        print ('False Numeric') if new_col.isnull().all() else ('True Numeric')
        return False if new_col.isnull().all() else True
    except (ValueError, TypeError):
        print('Not numeric in function')
        return False
    

def infer_data_types(file_path):
    """
    Infers data types for each column in a CSV or Excel file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        tuple: A tuple containing the pandas DataFrame and a dictionary
               of inferred data types for each column.
    """
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    inferred_types = {}
    for column in df.columns:
        flag = True
        try:
            # Check if the column contains dates
            df[column] = pd.to_datetime(df[column], format='%d/%m/%Y', errors='raise')
            # print('Found Date')
            inferred_types[column] = 'datetime64'
            flag = False
        except (ValueError, TypeError):
            # If not datetime, numeric conversion
            try:
                if is_numeric_column(df, column):
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    print('Printing this column')
                    print(df[column])
                # else:
                #     print('Nope, not numeric')
                #     pass
                # df[column] = pd.to_numeric(df[column])
               
                # print('Found Numeric')
            except:
                # print('Not Numeric but can be')
                pass
                
                
            if pd.api.types.is_numeric_dtype(df[column]):
                # print('Found Float')
                # inferred_types[column] = 'float64' if df[column].dtype == 'float' else 'int64'
                # df[column] = convert_to_int_float(df, column)
                all_values_end_with_zero_or_nan(df, column)
                inferred_types[column] = 'int64' if is_integer_column(df, column) else 'float64'
                flag = False
            elif df[column].nunique() < 0.5 * len(df):
                # Check for categorical data
                df[column] = df[column].astype('category')
                inferred_types[column] = 'category'
            elif pd.api.types.is_string_dtype(df[column]):
                # print('Found String')
                inferred_types[column] = 'string'
                flag = False 
            else:
                if flag:
                    # print('I am object')
                    print(df[column])
                    inferred_types[column] = 'object'
    
    return df, inferred_types

if __name__ == "__main__":
    
    df, inferred_types = infer_data_types("sample_data.csv")
    
    
    print(df.head())
    print(inferred_types)