import yfinance as yf
import pandas as pd
import numpy as np

# Fetch weekly Adjusted Close for SPY (dividend-adjusted)
df = yf.download("SPY", start="2010-01-01", end="2025-06-14", interval="1wk", auto_adjust=False)
df = df[["Adj Close"]].rename(columns={"Adj Close":"Price"})
df["Return"] = np.log(df["Price"]).diff()
df.to_csv("raw_data/spy_weekly_returns.csv")
