

import pandas as pd
from pytrends.request import TrendReq
import time
from datetime import datetime, timedelta
import sys
import random

def process_google_trends(
    terms: list,
    start_date: str,
    end_date: str,
    geo: str,
    output_path: str
) -> None:
    """
    Fetches Google Trends data for given terms between start_date and end_date,
    computes a weekly composite index, and writes Date + Anxiety to CSV.
    
    Args:
        terms (list): list of search terms, e.g. ['recession','market crash','unemployment']
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        geo (str): geographic region code ('US' or '' for worldwide)
        output_path (str): path for output CSV
    """
    try:
        # Initialize pytrends with a longer timeout
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
        
        # Split the date range into smaller chunks to avoid rate limiting
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        chunk_size = timedelta(days=15)  # 2-week chunks
        
        all_data = []
        current_start = start
        retry_count = 0
        max_retries = 1  # Only one retry attempt
        
        while current_start < end:
            try:
                current_end = min(current_start + chunk_size, end)
                timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
                
                print(f"Processing chunk: {timeframe}")
                
                # Build request payload
                pytrends.build_payload(terms, timeframe=timeframe, geo=geo)
                
                # Fetch interest over time
                df = pytrends.interest_over_time()
                if not df.empty:
                    if 'isPartial' in df.columns:
                        df = df.drop(columns=['isPartial'])
                    all_data.append(df)
                    print(f"Successfully processed chunk: {timeframe}")
                    retry_count = 0  # Reset retry count on success
                
                # Add random delay between requests (30-60 seconds)
                delay = random.uniform(30, 60)
                print(f"Waiting {delay:.1f} seconds before next request...")
                for _ in range(int(delay)):
                    try:
                        time.sleep(1)
                    except KeyboardInterrupt:
                        print("\nInterrupted during sleep. Saving progress...")
                        break
                
            except Exception as e:
                print(f"Error processing chunk {timeframe}: {str(e)}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    print(f"Max retries ({max_retries}) reached. Moving to next chunk.")
                    retry_count = 0
                    current_start = current_end + timedelta(days=1)
                    continue
                
                # Exponential backoff with jitter
                backoff_time = min(60, (2 ** retry_count) * 30 + random.uniform(0, 30))  # Shorter backoff
                print(f"Retrying in {backoff_time:.1f} seconds... (Attempt {retry_count + 1}/{max_retries})")
                
                try:
                    for _ in range(int(backoff_time)):
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nInterrupted during backoff. Moving to next chunk...")
                    retry_count = 0
                    current_start = current_end + timedelta(days=1)
                    continue
            
            current_start = current_end + timedelta(days=1)
        
        if not all_data:
            raise Exception("No data was collected from any time period")
        
        # Combine all chunks
        df = pd.concat(all_data)
        
        # Convert to weekly frequency by taking the mean
        df_weekly = df.resample('W').mean()
        
        # Compute composite Anxiety index as the average of all term columns
        df_weekly['Anxiety'] = df_weekly[terms].mean(axis=1)
        
        # Prepare final DataFrame
        df_processed = df_weekly[['Anxiety']].reset_index()
        df_processed.columns = ['Date', 'Anxiety']
        
        # Save to CSV
        df_processed.to_csv(output_path, index=False)
        print(f"Successfully processed and saved data to {output_path}")
        
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process Google Trends economic-anxiety terms into Date and Anxiety columns"
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        default=["recession", "market crash", "unemployment"],
        help="List of economic anxiety search terms"
    )
    parser.add_argument(
        "--start",
        default="2005-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        default="2011-04-03",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--geo",
        default="",
        help="Region code (e.g., 'US' or '' for worldwide)"
    )
    parser.add_argument(
        "--output",
        default="raw_data/google_anxiety_processed_1.csv",
        help="Path to save the processed CSV file"
    )
    args = parser.parse_args()

    process_google_trends(
        terms=args.terms,
        start_date=args.start,
        end_date=args.end,
        geo=args.geo,
        output_path=args.output
    )

