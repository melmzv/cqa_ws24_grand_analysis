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

    # Step 4: Match earnings announcements with stock returns
    final_matched = match_earnings_to_market_dates(final_filtered)

    # Step 5: Compute deviation statistics
    deviation_stats = compute_deviation_statistics(final_matched)

    # Save the prepared dataset
    final_matched.to_csv(cfg['prepared_wrds_ds2dsf_path'], index=False)
    final_matched.to_parquet(cfg['prepared_wrds_ds2dsf_parquet'], index=False)

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
    4. Removes rows where all four earnings announcement dates are identical.
    Also, identifies the years in `marketdate` where `ret = 0` is most frequent.
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

    # Remove rows where all four announcement dates are identical
    before_unique_filter = len(df_filtered)
    df_filtered = df_filtered[
        ~((df_filtered['item5901'] == df_filtered['item5902']) &
          (df_filtered['item5901'] == df_filtered['item5903']) &
          (df_filtered['item5901'] == df_filtered['item5904']))
    ]
    after_unique_filter = len(df_filtered)
    log.info(f"Dropped {before_unique_filter - after_unique_filter} rows where all four earnings announcements were identical.")

    # Count NaN values in the 'ret' column before replacing them
    nan_count = df_filtered['ret'].isna().sum()
    # Replace NaN values in the 'ret' column with 0
    df_filtered['ret'] = df_filtered['ret'].fillna(0)
    log.info(f"Replaced {nan_count} NaN values in 'ret' column with 0.")

    # Convert 'marketdate' to datetime format
    df_filtered['marketdate'] = pd.to_datetime(df_filtered['marketdate'], errors='coerce')

    # Extract year from 'marketdate'
    df_filtered['market_year'] = df_filtered['marketdate'].dt.year

    # Count rows where ret=0
    zero_ret_count = (df_filtered['ret'] == 0).sum()
    log.info(f"Number of rows where 'ret' = 0: {zero_ret_count}")

    # Identify years with the most occurrences of ret = 0
    zero_ret_by_year = df_filtered[df_filtered['ret'] == 0]['market_year'].value_counts()
    
    log.info("Top 5 years with the most occurrences of 'ret = 0':")
    log.info(zero_ret_by_year.head(5).to_string())

    # Drop the temporary year columns
    df_filtered = df_filtered.drop(columns=['year_5901', 'year_5902', 'year_5903', 'year_5904'])

    log.info(f"Final dataset size after filtering: {len(df_filtered)}")

    return df_filtered


def match_earnings_to_market_dates(df):
    """
    Match quarterly earnings announcement dates (item5901-4) to market trading dates per firm.
    Assign the closest available stock return (ret) for each earnings date.
    """

    log.info("Matching earnings announcements to valid trading days...")

    # Convert 'marketdate' to datetime
    df["marketdate"] = pd.to_datetime(df["marketdate"], errors="coerce")

    # Filter only necessary columns for efficiency
    market_data = df[["infocode", "marketdate", "ret"]].drop_duplicates()

    # Create a lookup dictionary {infocode: DataFrame with marketdate & ret}
    market_lookup = {
        infocode: group.set_index("marketdate")["ret"]
        for infocode, group in market_data.groupby("infocode")
    }

    # Initialize new columns for matched returns and days deviation
    for q in range(1, 5):
        df[f"q{q}_day_0"] = pd.NaT  # Matched market date
        df[f"q{q}_ret"] = None  # Matched return value
        df[f"q{q}_day_0_deviation"] = None  # Days deviation

    # Iterate over each row to match market dates per firm
    for idx, row in df.iterrows():
        infocode = row["infocode"]

        if infocode not in market_lookup:
            continue  # Skip if no market data is available

        for q in range(1, 5):
            earnings_date = row[f"item590{q}"]

            if pd.isna(earnings_date):
                continue  # Skip if earnings date is missing

            earnings_date = pd.to_datetime(earnings_date)

            # Find closest available market date in firm's data
            available_dates = market_lookup[infocode].index
            matched_date = available_dates[available_dates >= earnings_date].min()

            if pd.isna(matched_date):
                continue  # No valid market date found, leave NaN

            # Assign values to the new columns
            df.at[idx, f"q{q}_day_0"] = matched_date
            df.at[idx, f"q{q}_ret"] = market_lookup[infocode].loc[matched_date]
            df.at[idx, f"q{q}_day_0_deviation"] = (matched_date - earnings_date).days

    log.info("Earnings announcement matching complete.")
    return df


def compute_deviation_statistics(df):
    """
    Compute summary statistics for the deviation in event window alignment.
    """
    log.info("Computing deviation statistics...")

    deviation_stats = {
        f"q{q}_day_0_deviation": df[f"q{q}_day_0_deviation"].mean(skipna=True)
        for q in range(1, 5)
    }

    log.info(f"Average days deviation per quarter: {deviation_stats}")
    return deviation_stats


if __name__ == "__main__":
    main()