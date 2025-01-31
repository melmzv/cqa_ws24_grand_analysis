# We start by loading the libraries that we will use in this analysis.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import read_config, setup_logging

# Set up logging
log = setup_logging()

def main():
    log.info("Starting analysis ...")
    cfg = read_config('config/prepare_data_cfg.yaml')

    # Load datasets
    log.info("Loading computed BHR Event and BHR Annual datasets...")
    bhr_event_results = pd.read_csv(cfg["bhr_event_output_csv"])
    bhr_annual_results = pd.read_csv(cfg["bhr_annual_output_csv"])

    # Compute summary statistics
    df_summary = compute_summary_statistics(bhr_annual_results, bhr_event_results)



    log.info("Analysis complete.")

def compute_summary_statistics(bhr_annual_results, bhr_event_results):
    """
    Computes summary statistics (Mean, Median, Skewness, % Obs = 0, % Obs > 0) 
    for annual and earnings-announcement window returns.
    """
    log.info("Computing summary statistics for comparison with Table 1...")

    # **Annual Returns Summary**
    annual_stats = {
        "Category": "Calendar-Year Returns",
        "No. Obs.": len(bhr_annual_results),
        "Mean": np.mean(bhr_annual_results["BHR_Annual"]),
        "Median": np.median(bhr_annual_results["BHR_Annual"]),
        "Skewness": pd.Series(bhr_annual_results["BHR_Annual"]).skew(),
        "% Obs. = 0": np.mean(bhr_annual_results["BHR_Annual"] == 0) * 100,
        "% Obs. > 0": np.mean(bhr_annual_results["BHR_Annual"] > 0) * 100,
    }

    # **Earnings-Announcement Window Summary (for each quarter)**
    quarterly_stats = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        df_q = bhr_event_results[bhr_event_results["quarter"] == q]

        q_stats = {
            "Category": f"Earnings-Announcement Window Returns in {q}",
            "No. Obs.": len(df_q),
            "Mean": np.mean(df_q["BHR_3day"]),
            "Median": np.median(df_q["BHR_3day"]),
            "Skewness": pd.Series(df_q["BHR_3day"]).skew(),
            "% Obs. = 0": np.mean(df_q["BHR_3day"] == 0) * 100,
            "% Obs. > 0": np.mean(df_q["BHR_3day"] > 0) * 100,
        }

        quarterly_stats.append(q_stats)

    # Convert to DataFrame
    df_summary = pd.DataFrame([annual_stats] + quarterly_stats)

    log.info("Computed summary statistics:")
    log.info(df_summary.to_string())

    return df_summary

def plot_summary_statistics(df_summary):
    """
    Plots a bar chart comparing our computed summary statistics with Ball (2008) Table 1.
    """
    log.info("Plotting summary statistics for comparison with Ball’s Table 1...")

    fig, ax = plt.subplots(figsize=(10, 6))
    df_summary.set_index("Category")[["Mean", "Median", "Skewness"]].plot(kind="bar", ax=ax)
    
    plt.title("Comparison of Summary Statistics (Our Results vs. Ball Table 1)")
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save figure
    plot_path = "data/generated/summary_statistics_plot.png"
    plt.savefig(plot_path)
    log.info(f"Summary statistics plot saved to {plot_path}")

    plt.show()

if __name__ == "__main__":
    main()
