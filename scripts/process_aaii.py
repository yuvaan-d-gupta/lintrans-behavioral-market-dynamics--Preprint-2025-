

import pandas as pd

def process_aaii(input_path: str, output_path: str) -> None:
    # 1. Load the raw Excel file
    df = pd.read_excel(input_path)

    # 2. Ensure 'Date' column is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # 3. Compute net sentiment = Bullish - Bearish
    df['Sentiment'] = df['Bullish'] - df['Bearish']

    # 4. Select only the Date and Sentiment columns
    df_processed = df[['Date', 'Sentiment']].copy()

    # 5. Save the processed DataFrame to CSV
    df_processed.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process AAII sentiment data into Date and Sentiment columns"
    )
    parser.add_argument(
        "--input",
        default="raw_data/aaii_sentiment.xlsx",
        help="Path to the raw AAII Excel file",
    )
    parser.add_argument(
        "--output",
        default="raw_data/aaii_sentiment_processed.csv",
        help="Path to save the processed CSV file",
    )
    args = parser.parse_args()

    process_aaii(args.input, args.output)

# End of process_aaii.py
