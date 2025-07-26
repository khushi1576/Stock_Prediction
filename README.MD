## 📈 Stock Price Lookup & Prediction App

A Flask-based web application to **search and analyze** Indian and Global stocks by name. The app provides:

* Real-time **current stock price**
* **Historical data** (last 5 to 365 days)
* 📉 Close price charts
* 📈 **Future close price prediction** for next 30 days using Linear Regression
* High and low estimates from future predictions
* Yahoo Finance link for deep stock research

---

### 🗂️ Project Structure

```
stock-price/
│
├── static/
│   ├── stock_chart.png             # Historical chart image
│   ├── prediction_chart.png        # Prediction chart image
│
├── templates/
│   └── index.html                  # Frontend HTML template
│
├── app.py                          # Flask app with all logic
└── README.md                       # Project documentation
```

---

### 🚀 Features

* 🔍 **Search any stock by name** (e.g., Apple, TCS, Infosys, Tesla)
* 🌍 Supports **Indian & global stock exchanges**
* 📆 Choose from **5, 30, 90, 180, or 365** days of historical data
* 📊 View:

  * Open, High, Low, Close values
  * Current market price
* 📈 **Predict the next 30 days' stock prices**

  * Shows predicted chart
  * Highlights highest and lowest expected values
* 🔗 Direct link to Yahoo Finance for detailed company info

---

### 🧰 Tech Stack

* Python 3
* Flask
* yfinance (Yahoo Finance API)
* scikit-learn (for regression)
* matplotlib
* HTML/CSS (Jinja2 templating)

---

### 💻 Installation

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/stock-price.git
cd stock-price
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install flask yfinance matplotlib pandas scikit-learn
```

3. **Run the app**

```bash
python app.py
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

### 📸 Screenshots

![App Screenshot](static/stock_chart.png)

---

### 📂 Notes

* Ensure the `static/` folder exists before running the app.
* If you're deploying it publicly, turn `debug=True` to `False`.

---

### 📄 License

MIT License © 2025
