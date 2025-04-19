# 🧠 Disease Predictor with Real-Time and Long-Term Health Insights

This project is a hybrid health analytics pipeline that integrates machine learning-based long-term risk prediction and real-time anomaly detection using wearable data (e.g., from Apple Watch). The project leverages Python, Node-RED, and MQTT for seamless data flow and visualization.

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python >= 3.11
- Node.js (for Node-RED)
- node-red-dashboard module (for visualization of results in node-red)

Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
.
├── dataset/
│   └── heart.csv                  # Main dataset
├── preprocess/
│   └── preprocess.ipynb          # Jupyter notebook for preprocessing and model training
├── process/
│   ├── process.py                # Main ML pipeline and prediction script
│   └── process_demo.py           # Sample demo using test inputs
├── output/
│   ├── short_term_alert.json     # Real-time alerts (generated)
│   ├── node-red_flow.json        # Node-RED flow for UI/dashboard
│   └── Result2MQTT.py            # Publishes results to MQTT for dashboard display
├── requirements.txt
└── README.md
```

---

## 📊 Features

### 1. Data Preprocessing
- Cleans and encodes the heart disease dataset
- Normalizes continuous features using `StandardScaler`
- Visualizes data correlation and age-related risk trends

### 2. Long-Term Risk Prediction
- Trains a Random Forest classifier to predict heart disease risk
- Generates a binary risk output: "High Risk" or "Low Risk"
- Simulates time-based vital sign data for trend analysis

### 3. Real-Time Monitoring (with Apple Watch Integration)
- Ingests Apple Watch heart rate data (`heartrate_applewatch.csv`) or simulates it if unavailable
- Generates synthetic vitals (heart rate, O2 saturation, temperature, etc.)
- Computes user-specific baselines
- Detects anomalies using:
  - Clinical thresholds
  - Deviation from baseline using standard deviation

### 4. Output and Integration
- Exports real-time alerts as JSON
- Publishes predictions and alerts to Node-RED dashboard using MQTT
- Visual UI hosted at:  
  - `http://127.0.0.1:1880/` (Node-RED editor)  
  - `http://127.0.0.1:1880/ui` (Dashboard)

---

## 🧪 Run the Project

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Preprocessing and Modeling**
   ```bash
   cd preprocess
   jupyter notebook preprocess.ipynb
   # OR run all cells in VSCode
   ```

3. **Execute Prediction Pipelines**
   ```bash
   cd ../process
   python process.py
   python process_demo.py
   ```

4. **Launch Node-RED Dashboard**
   ```bash
   cd ../output
   node-red
   ```
   - Import the flow file `node-red_flow.json`
   - Open:
     - `http://127.0.0.1:1880/` (editor)
     - `http://127.0.0.1:1880/ui` (dashboard)

5. **Stream Data via MQTT**
   ```bash
   python Result2MQTT.py
   ```

---

## 📷 Sample Visualizations

- Correlation Heatmap
- Heart Disease Risk by Age Group
- Simulated Real-Time Heart Rate Trend
- Live Alerts Display with Baseline-Aware Anomaly Detection

---

## 📫 Contact

If you encounter any issues, please reach out to:  
**Team Coordinator: BAI Haoran**  
📧 hao-ran.bai@connect.polyu.hk

We’re happy to help and would like to hear your feedback!
