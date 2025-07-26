from flask import Flask, render_template, request
import yfinance as yf   #type:ignore
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def get_symbol_from_name(company_name):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, None
        results = response.json()
        if "quotes" in results and len(results["quotes"]) > 0:
            best_match = results["quotes"][0]
            return best_match["symbol"], best_match.get("longname", best_match.get("shortname", company_name))
    except:
        return None, None
    return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_path = None
    prediction_chart = None
    result = {}
    selected_days = "30"

    if request.method == 'POST':
        company = request.form['company']
        selected_days = request.form.get('days', '30')
        days = int(selected_days)
        symbol, full_name = get_symbol_from_name(company)

        if not symbol:
            result['error'] = "❌ Could not find a matching stock symbol."
        else:
            try:
                stock = yf.Ticker(symbol)
                hist_data = stock.history(period=f"{days}d", interval="1d")
                info = stock.info
                current_price = info.get("currentPrice", "N/A")

                if hist_data.empty:
                    result['error'] = "⚠️ No historical data available."
                else:
                    latest = hist_data.iloc[-1]
                    result.update({
                        "symbol": symbol,
                        "name": full_name,
                        "date": latest.name.date(),
                        "open": round(latest['Open'], 2),
                        "high": round(latest['High'], 2),
                        "low": round(latest['Low'], 2),
                        "close": round(latest['Close'], 2),
                        "current": current_price,
                        "yahoo_url": f"https://finance.yahoo.com/quote/{symbol}",
                        "days": days
                    })

                    # Plot Historical Chart
                    chart_path = 'static/stock_chart.png'
                    hist_data['Close'].plot(title=f"{full_name} - Last {days} Days", figsize=(8, 4))
                    plt.xlabel("Date")
                    plt.ylabel("Close Price")
                    plt.tight_layout()
                    plt.savefig(chart_path)
                    plt.close()

                    # Prediction using Linear Regression
                    hist_data = hist_data.reset_index()
                    hist_data['DateOrdinal'] = hist_data['Date'].map(pd.Timestamp.toordinal)
                    X = hist_data['DateOrdinal'].values.reshape(-1, 1)
                    y = hist_data['Close'].values

                    model = LinearRegression()
                    model.fit(X, y)

                    last_date = hist_data['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
                    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
                    predictions = model.predict(future_ordinals)

                    predicted_high = round(np.max(predictions), 2)
                    predicted_low = round(np.min(predictions), 2)

                    result['future_high'] = predicted_high
                    result['future_low'] = predicted_low

                    # Plot Prediction Chart
                    prediction_chart = 'static/prediction_chart.png'
                    plt.figure(figsize=(10, 4))
                    plt.plot(hist_data['Date'], y, label='Historical Close')
                    plt.plot(future_dates, predictions, label='Predicted (Next 30 Days)', linestyle='--')
                    plt.xlabel("Date")
                    plt.ylabel("Close Price")
                    plt.title(f"{full_name} - Close Price Prediction")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(prediction_chart)
                    plt.close()

            except Exception as e:
                result['error'] = f"⚠️ Failed to fetch data. Details: {e}"

    return render_template("index.html", result=result, chart=chart_path, prediction_chart=prediction_chart, selected_days=selected_days)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
