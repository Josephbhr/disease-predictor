

# 🧠 Disease Predictor: Long-Term Health Risk Forecasting System

Welcome to the **Disease Predictor** project — a smart health system that uses simulated patient data and machine learning to predict long-term cardiovascular risks. The project includes an interactive dashboard powered by **Node-RED**, real-time messaging via **MQTT**, and predictive modeling using **Random Forest**, **Logistic Regression**, **KNN**, and **Gaussian Processing**.

---

## ⚙️ Requirements

- **Python**: version **>= 3.11**
- **Node.js** and [**Node-RED**](https://nodered.org/) and **node-red-dashboard** moduleinstalled
- MQTT broker credentials (we use HiveMQ Cloud)

---

## 🚀 Getting Started

### 1. Install Python Dependencies

Open a terminal in the project root and run:

```bash
pip install -r requirements.txt
```

This installs all required Python packages.

---

### 2. Preprocess the Dataset

```bash
cd preprocess
```

You can either:

- Open `preprocess.py` in **VSCode** and click **"Run All"**, or  
- Run the script manually:

```bash
python preprocess.py
```

This step processes the heart disease dataset for model training.

---

### 3. Train & Run Prediction Models

```bash
cd ../process
```

Run the two scripts:

```bash
python process.py           # Trains models and outputs results
python process_demo.py      # Simulates a user and predicts their long-term health risk
```

Both models (Random Forest & Logistic Regression) will output risk classification and probabilities.

---

### 4. Launch Node-RED Dashboard

```bash
cd ../output
```

Start Node-RED:

```bash
node-red
```

- Open the Node-RED editor at: [http://127.0.0.1:1880](http://127.0.0.1:1880)
- Import the flow from:

```
output/node-red_flow.json
```

- Then visit your dashboard at:

[http://127.0.0.1:1880/ui](http://127.0.0.1:1880/ui)

> ⚠️ If port 1880 is occupied, Node-RED will use a different one. Check the terminal output to find the new URL.

---

### 5. Publish Predictions to MQTT

Still in the `/output` folder, run:

```bash
python Result2MQTT.py
```

This script sends the simulated user's prediction (including probabilities and risk factors) to the MQTT topic, so it can appear in real time on the Node-RED dashboard.

---

## 📂 Project Structure of Main Files for Testing

```
disease-predictor/
│
├── preprocess/           # Dataset preprocessing scripts
│   └── preprocess.py
│
├── process/              # ML training and simulated prediction
│   ├── process.py
│   └── process_demo.py
│
├── output/               # Node-RED flow, MQTT publishing
│   ├── Result2MQTT.py
│   └── node-red_flow.json
│
├── dataset/              # Original heart disease dataset
│   └── heart.csv
│
├── requirements.txt      # Python dependencies
└── README.md             # You are here!
```

---

## 📩 Contact

For any questions or assistance:

**Hao-Ran Bai**  
📧 hao-ran.bai@connect.polyu.hk

We’re happy to help and truly appreciate your feedback!

---

## ✅ Features

- ✅ Preprocessing of real-world heart disease data
- ✅ Random Forest & Logistic Regression prediction models
- ✅ Simulation of synthetic user health profiles
- ✅ MQTT-based real-time communication
- ✅ Node-RED dashboard for live vitals and long-term predictions

---

Thanks for exploring our project! 




 
