"""Summary statistics computed exclusively from individual test-life rows."""

import pandas as pd


SUMMARY_ROWS = ("Life_mean", "Life_median", "Life_std")


def life_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Discard previously calculated statistics, including legacy incorrect ones."""
    return metrics_df.drop(index=list(SUMMARY_ROWS), errors="ignore")


def add_life_statistics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Update summaries in place; std is the sample SD across test lives (ddof=1).

    A single life has an undefined sample SD, represented as NaN. Repeated calls
    produce the same result because existing summary rows are always excluded.
    """
    lives = life_rows(metrics_df)
    if lives.empty:
        raise ValueError("Cannot summarize metrics without individual test-life rows")
    summaries = (lives.mean(), lives.median(), lives.std(ddof=1))
    for label, values in zip(SUMMARY_ROWS, summaries):
        metrics_df.loc[label] = values.round(2)
    return metrics_df
