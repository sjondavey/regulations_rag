import logging
import os
import pandas as pd
from cryptography.fernet import Fernet


logger = logging.getLogger(__name__)

# Create custom log levels for the really detailed logs
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       

def load_parquet_data(path_to_file, decryption_key = ""):
    if not os.path.exists(path_to_file):
        msg = f"Could not find the file {path_to_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_parquet(path_to_file, engine='pyarrow')
    if decryption_key:
        fernet = Fernet(decryption_key)
        df['text'] = df['text'].apply(lambda x: fernet.decrypt(x.encode()).decode())
    return df

def save_parquet_data(df, path_to_file, decryption_key = ""):
    if not os.path.exists(path_to_file):
        msg = f"Could not find the file {path_to_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if decryption_key:
        fernet = Fernet(decryption_key)
        df['text'] = df['text'].apply(lambda x: fernet.encrypt(x.encode()).decode())
    df.to_parquet(path_to_file, engine = 'pyarrow')
    # but leave the column unchanged in the input df so the user can continue to use it
    if decryption_key:
        df['text'] = df['text'].apply(lambda x: fernet.decrypt(x.encode()).decode())


def append_parquet_data(path_to_file, original_df, decryption_key = ""):
    if path_to_file == "":
        return original_df

    tmp = load_parquet_data(path_to_file, decryption_key)

    return pd.concat([original_df, tmp], ignore_index = True)


def load_csv_data(path_to_file):
    """
    Loads data from a CSV file, ensuring no NaN values are present.

    Parameters:
    -----------
    path_to_file : str
        The path to the CSV file to be loaded.

    Returns:
    --------
    df : DataFrame
        The loaded DataFrame if the file exists and contains no NaN values.

    Raises:
    -------
    FileNotFoundError:
        If the specified file does not exist.
    ValueError:
        If the loaded DataFrame contains NaN values.
    """
    if not os.path.exists(path_to_file):
        msg = f"Could not find the file {path_to_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(path_to_file, sep="|", encoding="utf-8", na_filter=False)  

    # Check for NaN values in the DataFrame
    if df.isna().any().any():
        msg = f'Encountered NaN values while loading {path_to_file}. This will cause ugly issues with the get_regulation_detail method'
        logger.error(msg)
        raise ValueError(msg)
    return df

def append_csv_data(path_to_file, original_df):
    if path_to_file == "":
        return original_df

    tmp = load_csv_data(path_to_file)
    # data in the "_plus.csv" file contains an additional column "sections_referenced" which is only used to identify the rows that need to be updated when the manual changes
    if "sections_referenced" in tmp.columns:
        tmp.drop("sections_referenced", axis=1, inplace=True)

    return pd.concat([original_df, tmp], ignore_index = True)

def load_regulation_data_from_files(path_to_manual_as_csv_file, path_to_additional_manual_as_csv_file):
    df_regulations = load_csv_data(path_to_manual_as_csv_file)
    df_regulations = append_csv_data(path_to_additional_manual_as_csv_file, df_regulations)
    if df_regulations.isna().any().any():
        msg = f'Encountered NaN values while adding the two DataFrames together. This is caused because they dont have the same column names'
        logger.error(msg)
        raise ValueError(msg)
    return df_regulations

