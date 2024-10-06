# filename: ytd_gain.py
import yfinance as yf
import pandas as pd

# List of the 10 largest technology companies
tech_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'INTC']

# Create a dictionary to store the YTD gains
ytd_gains = {}

# Loop through each company and fetch its YTD gain
for company in tech_companies:
    ticker = yf.Ticker(company)
    hist = ticker.history(period="ytd")
    if not hist.empty:
        start_price = hist.iloc[0]['Close']
        end_price = hist.iloc[-1]['Close']
        ytd_gain = ((end_price - start_price) / start_price) * 100
        ytd_gains[company] = ytd_gain
    else:
        ytd_gains[company] = "No data available"

# Print the YTD gains
for company, gain in ytd_gains.items():
    print(f"{company}: {gain}%")