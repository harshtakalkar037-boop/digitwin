# 🤖 DigiTwin.AI — Predictive Maintenance Digital Twin

> **AI-powered industrial equipment monitoring system** that detects anomalies, estimates machine health, predicts Remaining Useful Life (RUL), and provides maintenance recommendations in real time.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![NumPy](https://img.shields.io/badge/NumPy-AI_Model-blue?logo=numpy)
![AI](https://img.shields.io/badge/AI-Autoencoder-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Overview

**DigiTwin.AI** is an intelligent digital twin system developed for predictive maintenance of industrial machines.

The system monitors three critical machine parameters:

* 🌡️ Temperature
* ⚙️ Load
* 📈 Vibration

Using a custom-built **Autoencoder Neural Network** implemented from scratch in NumPy, the system learns normal operating behavior and identifies anomalies using reconstruction error.

---

## 🧠 AI Model Architecture

```text
Input Layer (3 Features)
    ↓
Dense (8 Neurons)
    ↓
Dense (4 Neurons)
    ↓
Latent Space (2 Neurons)
    ↓
Dense (4 Neurons)
    ↓
Dense (8 Neurons)
    ↓
Output Layer (3 Features)
```

### Features Used

| Feature     | Description                    |
| ----------- | ------------------------------ |
| Temperature | Machine operating temperature  |
| Load        | Percentage load on the machine |
| Vibration   | Mechanical vibration level     |

---

## ✨ Key Features

* 🤖 Custom Autoencoder AI model (NumPy only)
* 📊 Real-time anomaly detection
* ❤️ Health score calculation (0–100%)
* ⏳ Remaining Useful Life (RUL) estimation
* 🔴 Warning and critical alerts
* 🛠️ Automated maintenance recommendations
* 📈 Training loss tracking
* 🔄 Live simulation modes
* 🌐 REST API with Flask
* ⚡ CORS-enabled backend for frontend integration

---

## 🧠 How It Works

1. Generate synthetic normal and anomalous machine data.
2. Train the autoencoder on normal data only.
3. Reconstruct incoming sensor data.
4. Compute Mean Squared Error (MSE).
5. Compare MSE against threshold.
6. Estimate health and RUL.
7. Return actionable maintenance recommendations.

---

## 📂 Project Structure

```text
DigiTwin.AI/
│── server.py          # Main Flask backend
│── README.md          # Project documentation
│── requirements.txt   # Python dependencies
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/digitwin-ai.git
cd digitwin-ai
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install flask flask-cors numpy
```

### 4. Run the Server

```bash
python server.py
```

---

## 🌐 API Endpoints

| Endpoint            | Method | Description                        |
| ------------------- | ------ | ---------------------------------- |
| `/`                 | GET    | Server status                      |
| `/status`           | GET    | Training status and metrics        |
| `/loss_log`         | GET    | Training loss values               |
| `/simulate`         | POST   | Analyze custom sensor input        |
| `/live/tick`        | GET    | Generate live simulated data       |
| `/live/mode/<mode>` | POST   | Change simulation mode             |
| `/live/history`     | GET    | Retrieve recent simulation history |

---

## 🧪 Example API Request

### POST `/simulate`

```json
{
  "temp": 85,
  "load": 78,
  "vib": 6.2
}
```

### Example Response

```json
{
  "temp": 85,
  "load": 78,
  "vib": 6.2,
  "mse": 0.0342,
  "health": 82,
  "rul": 145,
  "status": {
    "s": "NORMAL",
    "c": "green",
    "action": "Continue monitoring. No intervention needed."
  },
  "recs": [
    "All parameters nominal. No action required."
  ]
}
```

---

## 📊 Health Status Logic

| Health Score | Status          |
| ------------ | --------------- |
| 80–100%      | 🟢 Normal       |
| 60–79%       | 🟡 Mild Warning |
| 30–59%       | 🟠 Warning      |
| 0–29%        | 🔴 Critical     |

---

## 🔄 Live Simulation Modes

### `degrading`

Gradual deterioration over time.

### `spike`

Sudden temporary failures.

### `fluctuating`

Oscillating operating conditions.

---

## 📈 Training Details

| Parameter         | Value                   |
| ----------------- | ----------------------- |
| Normal Samples    | 800                     |
| Anomalous Samples | 150                     |
| Epochs            | 80                      |
| Batch Size        | 32                      |
| Learning Rate     | 0.005                   |
| Loss Function     | Mean Squared Error      |
| Optimizer         | Manual Gradient Descent |

---

## 🛠️ Technology Stack

* Python
* Flask
* Flask-CORS
* NumPy
* Threading

---

## 🎯 Use Cases

* Industrial predictive maintenance
* Smart factories
* Condition-based monitoring
* Digital twins
* Academic AI projects
* Hackathons and engineering competitions

---

## 🏆 Team Tech Titans

Developed by **Tech Titans** at **Pune Institute of Computer Technology (PICT), Pune**.

---

## 📸 Screenshots

Add your dashboard screenshots here:

```markdown
![Dashboard](images/dashboard.png)
```

---

## 📌 Future Enhancements

* Database integration
* MQTT/IoT sensor support
* Cloud deployment
* Advanced deep learning models
* Frontend dashboard
* Real-time alerts (email/SMS)

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork it
* 🛠️ Contribute improvements
* 📢 Share it with others

---

## 📬 Contact

**Harsh Takalkar**
📧 [harshtakalkar037@gmail.com](mailto:harshtakalkar037@gmail.com)

---

> “Predict failures before they happen.” 🚀
