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
    merged_dataset = merge_with_datastream(ws_expanded, ds2dsf)

    # Step 5: Select firms that meet sample criteria
    log.info("Selecting firms that meet the sample criteria (4 announcements per year)...")
    final_dataset = select_firms_for_sample(merged_dataset)

    # Step 5: Compute the Buy-and-Hold Returns (BHR)
    bhr_results = compute_eawr_bhr(final_dataset)

    # Step 6: Save only the BHR results
    save_bhr_results(bhr_results, cfg)

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
    If no valid trading day is found for Day 0, the entire event window (-3 to +3) is dropped.
    Finally, removes event windows (-3, -2, +2, +3) as they are no longer needed.
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

    # Identify where `ret = 0` for event windows -1, 0, +1
    zero_ret_rows = df_final[(df_final["event_window"].isin([-1, 0, 1])) & (df_final["ret"] == 0)]
    log.info(f"Identified {len(zero_ret_rows)} cases where `ret = 0` in key event windows (-1, 0, +1).")

    # List of infocode & year_ pairs where no valid trading day is found
    failed_rdq_infocode_pairs = []

    # **SHIFTING MECHANISM** - Adjusts Day 0 first, then Day -1 and Day +1 dynamically
    for index, row in zero_ret_rows.iterrows():
        new_date = row["event_date"]
        shifted = False  # Track if shifting was successful

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
                    shifted = True  # Mark as shifted
                    break  # Exit loop once a valid date is found

            else:
                log.warning(f"No valid trading day found for infocode {row['infocode']} on {row['event_date']}. Marking for full window removal.")
                failed_rdq_infocode_pairs.append((row["infocode"], row["year_"]))
                break  # Exit loop if no more valid dates exist

    # **NEW STEP: Remove full event windows (-3 to +3) if no valid trading day was found**
    if failed_rdq_infocode_pairs:
        log.warning(f"Removing full event windows for {len(failed_rdq_infocode_pairs)} earnings announcements with no valid trading day.")

        # Convert list to DataFrame
        failed_rdq_df = pd.DataFrame(failed_rdq_infocode_pairs, columns=["infocode", "year_"]).drop_duplicates()

        # Remove all rows associated with these failed announcements
        df_final = df_final.merge(failed_rdq_df, on=["infocode", "year_"], how="left", indicator=True)
        df_final = df_final[df_final["_merge"] == "left_only"].drop(columns=["_merge"])

    # **DROP EVENT WINDOWS -3, -2, +2, +3**
    df_final = df_final[df_final["event_window"].isin([-1, 0, 1])]
    log.info("Dropped event windows (-3, -2, +2, +3) as they are no longer needed.")

    # Drop unnecessary variables after merging
    drop_columns = ["region", "typecode", "dscode", "marketdate"]
    df_final = df_final.drop(columns=drop_columns, errors="ignore")
    log.info(f"Dropped unnecessary columns: {drop_columns}")

    # **CHECK FOR DUPLICATES**
    duplicate_rows = df_final[df_final.duplicated()]
    num_duplicate_rows = len(duplicate_rows)

    if num_duplicate_rows > 0:
        log.warning(f"Found {num_duplicate_rows} duplicate rows in the dataset. Displaying the first 5 duplicate rows:")
        log.warning(f"\n{duplicate_rows.head(5)}")  # Display first 5 duplicate rows

        # Optional: Remove duplicates
        df_final = df_final.drop_duplicates().reset_index(drop=True)
        log.info(f"Removed duplicate rows. New dataset size: {len(df_final)}")

    log.info(f"Final merged dataset after adjusting `ret = 0`. Observations: {len(df_final)}")
    return df_final


def select_firms_for_sample(df):
    """
    Filters dataset to retain firms with exactly four earnings announcements per year.
    Ensures all four announcements fall within the same calendar year and belong to unique quarters (Q1, Q2, Q3, Q4).
    """
    log.info("Selecting firms that meet the sample criteria (4 earnings announcements per year)...")

    # Ensure `rdq` is in datetime format
    df["rdq"] = pd.to_datetime(df["rdq"], errors="coerce")

    # Extract the announcement year from `rdq`
    df["rdq_year"] = df["rdq"].dt.year

    # Keep only observations where event_window = 0 (earnings announcement day)
    df_event_0 = df[df["event_window"] == 0].copy()

    # Count unique quarters per firm-year where event_window == 0
    firm_rdq_counts = df_event_0.groupby(["infocode", "rdq_year"])["quarter"].nunique().reset_index()

    # Identify firms that have exactly 4 unique quarters (Q1, Q2, Q3, Q4)
    valid_firms = firm_rdq_counts[firm_rdq_counts["quarter"] == 4]


    # Merge back with the main dataset to keep only these firms
    df_filtered = df.merge(valid_firms[["infocode", "rdq_year"]], on=["infocode", "rdq_year"], how="inner")

    # Print the total number of unique firms after filtering
    unique_firms_after = df_filtered["infocode"].nunique()
    log.info(f"Total unique firms after filtering: {unique_firms_after}")

    log.info(f"Retained {unique_firms_after} firms meeting sample criteria.")

    return df_filtered

def compute_eawr_bhr(df):
    """
    Computes the Earnings Announcement Window Return (EAWR) as the 
    buy-and-hold return (BHR) over the three-day event window (-1,0,+1).
    Skips calculations for missing event windows.
    """
    log.info("Computing Earnings Announcement Window Returns (3-day BHR)...")

    # Ensure dataset is sorted properly
    df = df.sort_values(by=["infocode", "rdq", "event_window"])

    # Group by firm and earnings announcement date
    bhr_results = []
    
    for (infocode, rdq), group in df.groupby(["infocode", "rdq"]):
        # Ensure all required event windows are present
        if set(group["event_window"]) == {-1, 0, 1}:  
            try:
                ret_neg1 = group.loc[group["event_window"] == -1, "ret"].values[0]
                ret_0 = group.loc[group["event_window"] == 0, "ret"].values[0]
                ret_1 = group.loc[group["event_window"] == 1, "ret"].values[0]

                # Compute BHR_3day
                bhr_3day = (1 + ret_neg1) * (1 + ret_0) * (1 + ret_1) - 1

                # Append results
                bhr_results.append({"infocode": infocode, "rdq": rdq, "BHR_3day": bhr_3day})
            except Exception as e:
                log.warning(f"Skipping {infocode} on {rdq} due to missing data: {e}")
    
    # Convert to DataFrame
    df_bhr = pd.DataFrame(bhr_results)

    log.info(f"Computed {len(df_bhr)} earnings announcement window returns.")
    return df_bhr


def save_bhr_results(df_bhr, cfg):
    """
    Saves the computed BHR_3day dataset to CSV and Parquet formats.
    """
    log.info("Saving BHR dataset...")

    # Keep only relevant columns
    df_bhr_filtered = df_bhr[["infocode", "rdq", "BHR_3day"]]

    # Save dataset
    bhr_csv_path = cfg["bhr_output_csv"]
    bhr_parquet_path = cfg["bhr_output_parquet"]
    
    df_bhr_filtered.to_csv(bhr_csv_path, index=False)
    df_bhr_filtered.to_parquet(bhr_parquet_path, index=False)

    log.info(f"BHR dataset saved to {bhr_csv_path} (CSV) and {bhr_parquet_path} (Parquet).")

if __name__ == "__main__":
    main()