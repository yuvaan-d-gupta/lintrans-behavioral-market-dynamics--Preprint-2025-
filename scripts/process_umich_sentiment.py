

import pandas as pd
import argparse

def process_umich(input_path: str, output_path: str, method: str) -> None:
    """
    Reads monthly Consumer Sentiment with 'Month','Year','Index', resamples to weekly,
    and writes Date + Sentiment to CSV.

    Args:
        input_path: path to raw monthly CSV (skip first row title)
        output_path: path to save processed weekly CSV
        method: 'ffill' or 'interpolate'
    """
    # Load raw CSV, skip the title row
    df_raw = pd.read_csv(input_path, skiprows=1)
    
    # Rename columns and drop any empty/unnamed column
    df_raw.columns = ['Month', 'Year', 'Sentiment', 'Unused']
    df = df_raw[['Month', 'Year', 'Sentiment']].copy()
    
    # Construct a datetime 'Date' at month-start
    df['Date'] = pd.to_datetime(
        df['Year'].astype(int).astype(str) + '-' +
        df['Month'].astype(int).astype(str).str.zfill(2) + '-01'
    )
    df = df[['Date', 'Sentiment']].set_index('Date').sort_index()
    
    # Create weekly index
    weekly_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W')
    
    # Reindex and resample
    df_weekly = df.reindex(weekly_index)
    if method == 'ffill':
        df_weekly['Sentiment'] = df_weekly['Sentiment'].ffill()
    elif method == 'interpolate':
        df_weekly['Sentiment'] = df_weekly['Sentiment'].interpolate(method='time')
    else:
        raise ValueError("Method must be 'ffill' or 'interpolate'")
    
    # Prepare output
    df_out = df_weekly.reset_index()
    df_out.columns = ['Date', 'Sentiment']
    
    # Save CSV
    df_out.to_csv(output_path, index=False)
    print(f"Processed UMich sentiment saved to {output_path} (method={method})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process UMich Consumer Sentiment (monthly) to weekly"
    )
    parser.add_argument(
        "--input",
        default="raw_data/sca-table1-on-2025-Jun-16.csv",
        help="Path to raw UMich monthly CSV"
    )
    parser.add_argument(
        "--output",
        default="raw_data/umich_sentiment_weekly.csv",
        help="Path to save weekly CSV"
    )
    parser.add_argument(
        "--method",
        choices=['ffill', 'interpolate'],
        default='ffill',
        help="Resampling method"
    )
    args = parser.parse_args()
    process_umich(args.input, args.output, args.method)

