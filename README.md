# 📉 Market Anomaly Detection (Unsupervised ML)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Isolation%20Forest-orange)
![Status](https://img.shields.io/badge/Status-Working-green)

An unsupervised machine learning system to detect anomalies in financial market data.  
The project identifies unusual market behavior (price and volume spikes) that may indicate abnormal or risky market conditions.

---

## 🚀 Overview

This project implements **unsupervised anomaly detection** on financial market data using **Isolation Forest**.

Unlike traditional crash prediction models, this system:
- Does **not** rely on labeled crash data
- Learns normal market behavior automatically
- Flags extreme deviations as anomalies

---

## 🏗 Key Features

- Detects anomalies in **price and volume patterns**
- Uses **Isolation Forest** (industry-standard anomaly detection algorithm)
- No labeled data required
- Trained model saved for reuse
- Easily extendable to real-time or API-based systems

---

## ⚙ Tech Stack

- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Model Saving:** Joblib  

---

## 📂 Project Structure

```text
market-anomaly-detection/
├── data/
│   └── FinancialMarketData.csv
├── models/
│   └── isolation_forest.pkl
├── anomaly_detection.py
├── requirements.txt
├── README.md
```

---

## 📊 Dataset

The dataset contains numerical market features such as:

```text
Open, High, Low, Close, Volume
```

The model learns normal market behavior from these features and flags unusual patterns (e.g., sudden spikes in price or volume).

> ⚠️ The dataset is not included in the repository and must be added manually.

---

## 📝 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shashika8088/market-anomaly-detection.git
cd market-anomaly-detection
```

---

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
```

---

### 4️⃣ Add Dataset

Create the file:
```text
data/FinancialMarketData.csv
```

Example:
```csv
Open,High,Low,Close,Volume
100,105,99,102,120000
102,108,101,107,150000
150,180,140,175,900000
300,350,290,340,2000000
```

---

### 5️⃣ Run Anomaly Detection
```bash
python anomaly_detection.py
```

---

## ✅ Output

- Console output showing **Normal vs Anomaly** data points
- Trained model saved at:

```text
models/isolation_forest.pkl
```

---

## 📌 How It Works

1. Market data is scaled using standardization  
2. Isolation Forest isolates rare patterns  
3. Points isolated quickly are marked as anomalies  

Output labels:
- **Normal**
- **Anomaly**

---

## 🎯 Future Improvements

- 📈 Visualize anomalies using Matplotlib
- ⏱ Add real-time market data ingestion
- 🌐 Deploy as a FastAPI service
- 📊 Extend to multivariate time-series data
- 🧠 Compare with LOF and One-Class SVM