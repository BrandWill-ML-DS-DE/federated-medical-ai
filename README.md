# 🏥 Privacy-Preserving Federated Medical AI

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated_Learning-orange)](https://flower.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)

A **production-grade Federated Learning (FL) system** designed for collaborative medical diagnosis.  
This framework enables multiple institutions (hospitals) to train a **global AI model** on sensitive patient data **without sharing raw records**, using **Differential Privacy (DP)** to guarantee anonymity.

---

# 🧠 System Architecture

This system demonstrates a **secure distributed training pipeline** for **non-IID medical datasets** where hospitals contain different patient demographics.

```
                ┌────────────────────┐
                │ Federated Server   │
                │  (Flower + MLflow) │
                └─────────┬──────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
 ┌────────────┐   ┌────────────┐   ┌────────────┐
 │ Hospital A │   │ Hospital B │   │ Hospital C │
 │ PyTorch +  │   │ PyTorch +  │   │ PyTorch +  │
 │ Opacus DP  │   │ Opacus DP  │   │ Opacus DP  │
 └──────┬─────┘   └──────┬─────┘   └──────┬─────┘
        │                │                │
   Local Training   Local Training   Local Training
        │                │                │
        └──── Model Updates (No Raw Data) ┘

            Aggregation via FedAvg
```

---

# 🔄 Federated Training Workflow

1. **Server initializes a global model**
2. **Hospitals download the model weights**
3. Each hospital:
   - trains locally on private patient data
   - applies **Differential Privacy via Opacus**
4. Clients send **noisy gradient updates**
5. Server aggregates updates using **FedAvg**
6. Updated model redistributed to clients
7. Process repeats for **N communication rounds**

---

# 🔐 Privacy Protection

The training loop integrates **Opacus Differential Privacy** to protect patient information.

Key mechanisms:

- **Gradient Clipping**
- **Gaussian Noise Injection**
- **Privacy Budget Tracking**

Example configuration:

```
ε (epsilon): 0.95
δ (delta): 1e-5
Noise Multiplier: 1.1
Gradient Clip Norm: 1.0
```

Lower **ε** indicates **stronger privacy guarantees**.

---

# 🛠 Tech Stack

| Component | Technology | Role |
|-----------|------------|------|
| Federated Learning | Flower (flwr) | Distributed orchestration |
| Deep Learning | PyTorch | Model training |
| Privacy | Opacus | Differential Privacy |
| Experiment Tracking | MLflow | Logging & metrics |
| Deployment | FastAPI + Uvicorn | Inference API |
| Containerization | Docker | Reproducible environments |

---

# 📂 Project Structure

```
federated-medical-ai
│
├── client.py        # Federated client (hospital node)
├── server.py        # Federated aggregation server
├── data.py          # Non-IID dataset simulation
├── model.py         # PyTorch model architecture
├── api.py           # FastAPI inference API
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

## 1️⃣ Installation

```bash
git clone https://github.com/your-username/federated-medical-ai.git
cd federated-medical-ai

pip install -r requirements.txt
```

---

## 2️⃣ Start the Federated Server

```bash
python server.py
```

---

## 3️⃣ Launch Hospital Clients

Open separate terminals for each client:

```bash
python client.py 1
python client.py 2
python client.py 3
```

This simulates **three hospital nodes** participating in training.

---

## 4️⃣ Deploy the Global Model

After training completes:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The model will now be available as an API.

---

# 📡 Inference API

### Endpoint

```
POST /predict
```

### Example Request

```json
{
  "age": 65,
  "blood_pressure": 140,
  "cholesterol": 220,
  "glucose": 150
}
```

### Example Response

```json
{
  "diagnosis": "High Risk",
  "probability": 0.87
}
```

---

# 📊 Evaluation Metrics

| Metric | Value |
|------|------|
| Target Accuracy | ~85–92% |
| Privacy Budget (ε) | < 1.0 |
| Communication Rounds | 30 |

---

# 🧪 Non-IID Data Simulation

The `data.py` module simulates **real-world hospital data silos**, where:

- patient demographics vary across hospitals
- feature distributions differ
- training data is **heterogeneous**

This tests the ability of the global model to **generalize across institutions**.

---

# 🔒 Threat Model

The system protects against:

- Membership inference attacks
- Data leakage through gradients
- Cross-institution data exposure

Future work will address:

- Malicious clients
- Model poisoning attacks
- Byzantine failures

---

# 🗺 Future Roadmap

- [ ] **Secure Aggregation (SecAgg)** using cryptographic multi-party computation  
- [ ] **Medical Imaging Support** with 3D CNNs for MRI/CT scans  
- [ ] **FedProx Strategy** for heterogeneous client environments  
- [ ] **Client Trust Scoring** to defend against adversarial nodes  

---

# ⚠ Disclaimer

This project is a **technical demonstration**.

Although it integrates **Differential Privacy**, it is **not HIPAA-compliant** and should **not be used for real clinical diagnosis** without further security review and regulatory compliance.

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Dr. Brandon Williams
