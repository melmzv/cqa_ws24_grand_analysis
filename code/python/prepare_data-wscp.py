# --- Header -------------------------------------------------------------------
# Prepare the pulled data for further analysis as per Task 3 requirements
#
# (C) Melisa Mazaeva - See LICENSE file for details
# ------------------------------------------------------------------------------

import pandas as pd
from utils import read_config, setup_logging

log = setup_logging()

def main():
    log.info("Preparing data for analysis ...")
    cfg = read_config('config/prepare_data_cfg.yaml')

    # Load the pulled datasets
    ws_stock = pd.read_csv(cfg['worldscope_sample_save_path_csv'])
    link_ds_ws = pd.read_csv(cfg['link_ds_ws_save_path_csv'])
    ds2dsf = pd.read_csv(cfg['datastream_sample_save_path_csv'])
    
    # Step 1: Merge Worldscope with the Linking Table
    log.info("Merging Worldscope with Linking Table...")
    ws_link_merged = merge_worldscope_link(ws_stock, link_ds_ws)

    # Step 2: Merge the previous with Datastream data
    log.info("Merging merged data with Datastream...")
    final_merged = merge_with_datastream(ws_link_merged, ds2dsf)

    # Step 3: Filter for valid earnings announcement dates
    log.info("Filtering firm-year observations based on earnings announcement criteria...")
    final_filtered = filter_valid_earnings(final_merged)

    # Save the prepared dataset
    final_filtered.to_csv(cfg['prepared_wrds_ds2dsf_path'], index=False)
    final_filtered.to_parquet(cfg['prepared_wrds_ds2dsf_parquet'], index=False)

    log.info(f"Prepared data saved to {cfg['prepared_wrds_ds2dsf_path']} (CSV)")
    log.info(f"Prepared data saved to {cfg['prepared_wrds_ds2dsf_parquet']} (Parquet)")

    log.info("Preparing data for analysis ... Done!")


def merge_worldscope_link(ws_stock, link_ds_ws):
    """
    Merge Worldscope stock data with the linking table.
    Uses `code` (QA ID for Worldscope) to join with the linking table.
    """
    ws_link_merged = ws_stock.merge(link_ds_ws, left_on="code", right_on="code", how="inner")
    log.info(f"Merged Worldscope and Linking Table. Observations: {len(ws_link_merged)}")
    return ws_link_merged


def merge_with_datastream(ws_link_merged, ds2dsf):
    """
    Merge the Worldscope-linked dataset with Datastream.
    Uses `infocode` (QA ID for Datastream) as the linking key.
    """
    final_merged = ws_link_merged.merge(ds2dsf, on="infocode", how="inner")
    log.info(f"Merged with Datastream. Final observations: {len(final_merged)}")
    return final_merged


def filter_valid_earnings(df):
    """
    Filters out firm-year observations where:
    1. Any of the earnings announcement dates (items 5901-5904) are missing.
    2. All four earnings announcements must fall within the same calendar year.
    3. All four earnings announcement dates must be in the year **≤2023**.
    """
    initial_count = len(df)

    # Drop rows where any of the quarterly earnings announcement dates are missing
    df_filtered = df.dropna(subset=['item5901', 'item5902', 'item5903', 'item5904']).copy()
    after_na_drop = len(df_filtered)
    log.info(f"Dropped {initial_count - after_na_drop} rows due to missing earnings announcement dates.")

    # Convert date columns to datetime format
    date_columns = ['item5901', 'item5902', 'item5903', 'item5904']
    for col in date_columns:
        df_filtered[col] = pd.to_datetime(df_filtered[col], format="%m/%d/%y", errors='coerce')

    # Extract year from each earnings announcement
    df_filtered['year_5901'] = df_filtered['item5901'].dt.year
    df_filtered['year_5902'] = df_filtered['item5902'].dt.year
    df_filtered['year_5903'] = df_filtered['item5903'].dt.year
    df_filtered['year_5904'] = df_filtered['item5904'].dt.year

    # Ensure all four announcements fall within the same calendar year
    df_filtered = df_filtered[
        (df_filtered['year_5901'] == df_filtered['year_5902']) &
        (df_filtered['year_5901'] == df_filtered['year_5903']) &
        (df_filtered['year_5901'] == df_filtered['year_5904'])
    ]
    after_year_check = len(df_filtered)
    log.info(f"Dropped {after_na_drop - after_year_check} rows due to earnings announcements spanning multiple years.")

    # Remove firm-year observations where any date is beyond 2023
    df_filtered = df_filtered[
        (df_filtered['year_5901'] <= 2023) &
        (df_filtered['year_5902'] <= 2023) &
        (df_filtered['year_5903'] <= 2023) &
        (df_filtered['year_5904'] <= 2023)
    ]
    after_2023_check = len(df_filtered)
    log.info(f"Dropped {after_year_check - after_2023_check} rows due to earnings announcements beyond 2023.")

    # Count NaN values in the 'ret' column before replacing them
    nan_count = df_filtered['ret'].isna().sum()
    # Replace NaN values in the 'ret' column with 0
    df_filtered['ret'] = df_filtered['ret'].fillna(0)
    log.info(f"Replaced {nan_count} NaN values in 'ret' column with 0.")

    # Count rows where ret=0
    zero_ret_count = (df_filtered['ret'] == 0).sum()
    log.info(f"Number of rows where 'ret' = 0: {zero_ret_count}")

    # Drop the temporary year columns
    df_filtered = df_filtered.drop(columns=['year_5901', 'year_5902', 'year_5903', 'year_5904'])

    log.info(f"Final dataset size after filtering: {len(df_filtered)}")

    return df_filtered





if __name__ == "__main__":
    main()