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

    # Step 2: Pivot dataset to long format
    log.info("Pivoting merged dataset to long format...")
    ws_long = pivot_longer_earnings(ws_link_merged)

    # Step 3: Expand dataset for event windows (-1, 0, +1 days)
    log.info("Expanding dataset for event windows...")
    ws_expanded = expand_event_window(ws_long)

    # Step 4: Merge with Datastream stock returns
    log.info("Merging expanded dataset with Datastream stock returns...")
    final_dataset = merge_with_datastream(ws_expanded, ds2dsf)

    # Save the final dataset
    final_dataset.to_csv(cfg['prepared_wrds_ds2dsf_path'], index=False)
    final_dataset.to_parquet(cfg['prepared_wrds_ds2dsf_parquet'], index=False)

    log.info(f"Final dataset saved to {cfg['prepared_wrds_ds2dsf_path']} (CSV) and {cfg['prepared_wrds_ds2dsf_parquet']} (Parquet)")

    log.info("Preparing data for analysis ... Done!")


def merge_worldscope_link(ws_stock, link_ds_ws):
    """
    Merge Worldscope stock data with the linking table.
    Uses `code` (QA ID for Worldscope) to join with the linking table.
    Keeps only the relevant columns: year_, item6105, item5901, item5902, item5903, item5904, infocode.
    """
    # Merge datasets
    ws_link_merged = ws_stock.merge(link_ds_ws, left_on="code", right_on="code", how="inner")
    log.info(f"Merged Worldscope and Linking Table. Observations: {len(ws_link_merged)}")

    # Keep only relevant columns
    selected_columns = ["year_", "item6105", "item5901", "item5902", "item5903", "item5904", "infocode"]
    ws_link_merged = ws_link_merged[selected_columns]
    log.info(f"Filtered merged dataset to keep only relevant columns: {selected_columns}")

    return ws_link_merged

def pivot_longer_earnings(ws_link_merged):
    """
    Transforms the dataset from wide to long format, making each earnings announcement its own row.
    """
    log.info("Pivoting dataset to longer format...")

    # Pivot longer to have one earnings announcement per row
    ws_long = ws_link_merged.melt(
        id_vars=["year_", "item6105", "infocode"],  # Keep these as identifiers
        value_vars=["item5901", "item5902", "item5903", "item5904"],  # Pivot these columns
        var_name="quarter", 
        value_name="rdq"
    )

    # Map quarter names for clarity
    quarter_mapping = {
        "item5901": "Q1",
        "item5902": "Q2",
        "item5903": "Q3",
        "item5904": "Q4"
    }
    ws_long["quarter"] = ws_long["quarter"].map(quarter_mapping)

    # Log transformation
    log.info(f"Pivoted dataset. New number of rows: {len(ws_long)}")

    return ws_long

def expand_event_window(df):
    """
    Expands dataset by adding -3 to +3 day event windows for each earnings announcement.
    If `ret = 0` on Day 0, shift `event_date` to the next available trading day.
    """
    log.info("Expanding dataset to include extended event windows (-3 to +3 days)...")

    # Ensure rdq is properly parsed as datetime
    df["rdq"] = pd.to_datetime(df["rdq"], format="%m/%d/%y", errors="coerce")

    # Log the number of NaT values before expansion
    num_nat = df["rdq"].isna().sum()
    if num_nat > 0:
        log.warning(f"Found {num_nat} missing or unconvertible 'rdq' values. Skipping these rows.")

    # Drop NaT values to avoid issues in expansion
    df = df.dropna(subset=["rdq"]).copy()

    # Define the extended event window offsets (-3 to +3)
    offsets = [-3, -2, -1, 0, 1, 2, 3]

    # Generate new rows efficiently using pandas repeat + offsets
    df_expanded = df.loc[df.index.repeat(len(offsets))].reset_index(drop=True)
    df_expanded["event_window"] = offsets * (len(df))  # Apply offsets
    df_expanded["event_date"] = df_expanded["rdq"] + pd.to_timedelta(df_expanded["event_window"], unit="D")

    log.info(f"Expanded dataset. New number of rows: {len(df_expanded)}")
    return df_expanded


def merge_with_datastream(df_expanded, ds2dsf):
    """
    Merge expanded dataset with Datastream stock returns using `infocode` and `event_date`,
    then adjust `event_date` for missing stock returns (`ret = 0`), ensuring continuous shifting.
    """
    log.info("Merging with Datastream stock returns...")

    # Ensure marketdate is in datetime format
    ds2dsf["marketdate"] = pd.to_datetime(ds2dsf["marketdate"], format="%m/%d/%y", errors="coerce")

    # Merge on `infocode` and `event_date` = `marketdate`
    df_final = df_expanded.merge(ds2dsf, left_on=["infocode", "event_date"], right_on=["infocode", "marketdate"], how="left")

    # Check for unmatched event dates (missing stock return data)
    missing_ret_count = df_final["ret"].isna().sum()
    log.warning(f"{missing_ret_count} rows have missing stock return data. These will be removed.")

    # Drop rows where `ret` is missing (i.e., the merge was not possible)
    df_final = df_final.dropna(subset=["ret"]).copy()

    # Identify where `ret = 0` for event window 0
    zero_ret_rows = df_final[(df_final["event_window"] == 0) & (df_final["ret"] == 0)]
    log.info(f"Identified {len(zero_ret_rows)} cases where `ret = 0` on Day 0.")

    # Shift `event_date` continuously until a valid `ret != 0` is found
    for index, row in zero_ret_rows.iterrows():
        new_date = row["event_date"]

        while True:
            # Get the next available trading date with `ret != 0`
            possible_dates = df_final[
                (df_final["infocode"] == row["infocode"]) & 
                (df_final["event_date"] > new_date) & 
                (df_final["ret"] != 0)
            ].sort_values(by="event_date")

            if not possible_dates.empty:
                new_date = possible_dates.iloc[0]["event_date"]  # Update event_date
                new_ret = possible_dates.iloc[0]["ret"]

                # Ensure `ret != 0`
                if new_ret != 0:
                    df_final.at[index, "event_date"] = new_date
                    df_final.at[index, "ret"] = new_ret
                    break  # Exit loop once a valid date is found

            else:
                log.warning(f"No valid trading day found for infocode {row['infocode']} on {row['event_date']}. Keeping original.")
                break  # Exit loop if no more valid dates exist

    # Drop unnecessary variables after merging
    drop_columns = ["region", "typecode", "dscode", "marketdate"]
    df_final = df_final.drop(columns=drop_columns, errors="ignore")
    log.info(f"Dropped unnecessary columns: {drop_columns}")

    log.info(f"Final merged dataset after adjusting `ret = 0`. Observations: {len(df_final)}")
    return df_final


'''
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
    return deviation_stats'''


if __name__ == "__main__":
    main()