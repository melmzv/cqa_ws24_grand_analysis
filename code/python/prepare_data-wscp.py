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

    # Save the prepared dataset
    final_merged.to_csv(cfg['prepared_wrds_ds2dsf_path'], index=False)
    final_merged.to_parquet(cfg['prepared_wrds_ds2dsf_parquet'], index=False)

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



if __name__ == "__main__":
    main()