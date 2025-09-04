# FL-IDS: Federated Learning Intrusion Detection System for Drone Swarms

This repository contains the code and experimental artifacts for **FL-IDS**, a hybrid deep learning-based Intrusion Detection System (IDS) integrated with a Federated Learning (FL) framework. The system is designed to detect and mitigate network-level attacks such as DDoS in drone swarms operating in adversarial edge environments like AERPAW.

---

## Overview

The project implements a **hybrid CNN–LSTM–GRU model** trained on a customized dataset combining:
- The [Edge-IIoT Dataset](https://www.kaggle.com/datasets/sibasispradhan/edge-iiotset-dataset)
- Real-world DDoS and SQL injection attack traffic collected on the **AERPAW testbed**

The IDS is trained both in **centralized** and **federated settings**, and evaluated under normal, mildly congested (33%), and severely congested (66%) network scenarios.


## Dataset Details

### Edge-IIoT Dataset
- Protocols: ARP, ICMP, HTTP, TCP, UDP, DNS, MQTT, Modbus
- Labels: Binary (Attack/Normal), Multi-class (15+ attack types)
- Feature engineering includes label encoding, one-hot encoding, chi-squared feature selection, and scaling.

### AERPAW Attack Dataset
- Captured attacks: `DDoS_TCP`, `DDoS_UDP`, `DDoS_ICMP`, `DDoS_HTTP`
- Preprocessing includes categorical encoding, duplicate removal, and merge with Edge-IIoT.

---

## Model Architecture

- **Conv1D + MaxPooling**: Feature extraction
- **LSTM + GRU**: Captures long- and short-term temporal patterns
- **Dense + Dropout**: Regularized classification layer

### Configuration:
| Parameter              | Value           |
|------------------------|-----------------|
| Epochs                 | 6               |
| Batch Size             | 32              |
| Optimizer              | Adam            |
| Loss (Binary)          | Binary CrossEntropy |
| Loss (Multiclass)      | Sparse Categorical CrossEntropy |
| Output Activation      | Sigmoid / Softmax |
| Sequence Shape         | (samples, time_steps, features) |

---

## Experiments

### Goals:
- Detect and Mitigate DDoS (network-level)
- Evaluate performance under 33% and 66% attack intensities

### Environment:
- **Testbed Setup**
- Server: A compute node (VM/LAptop machine)
- Clients: Starling VOXL2 drone and Raspberry Pi 3
- Network tools: `iperf3`, `tc`

---

## Evaluation Metrics

| Metric Type   | Metrics                        |
|---------------|--------------------------------|
| Model         | Accuracy, Precision, Recall, F1 |
| System        | CPU usage, Training Time |
| Network       | Bandwidth, RTT, Packet Loss, Attack Recovery Time |

---z

## Results Snapshot

- **Binary Classification Accuracy (FL):** ~0.98
- **Multiclass Precision/Recall (centralized):** ~0.94
- **DDoS Traffic Detection:** >94% accuracy under realistic conditions
- **Federated Recovery:** Maintained accuracy even with 1–3 poisoned nodes

---

# How to Run

### Setup and Install dependencies
```bash
# Clone repository
git clone https://github.com/arculus-zt/arculus-fl 

# Install dependencies
pip install -r requirements.txt
```

### Centralized Training
#### Run centralized model
* For centralized binary classification
```bash
python3 centralized/binary_classification.py
```
* For centralized multiclass classification
```bash
python3 centralized/multiclass_classification.py
```

### Federated Training
#### Choose the directory which model you wanna train (**binary**/**multiclass**)
```bash
cd federated/binary #for binary classification
```
or
```bash
cd federated/multiclass #for multiclass classification
```
##### Launch the server
```bash
python3 server.py
```

#### Launch federated clients (on different terminals or nodes)
```bash
python3 federated/client.py -i 1
```
here **i** is the ID of the individual FL participant node client. For the second client, run the same script with `-i 2`


# DDoS Attack Generation

## Controlled Intensified DDoS using Iperf3

This command uses `iperf3` to measure high-throughput network performance between a client and server. It's configured to simulate a sustained, high-bandwidth data transfer using multiple parallel streams.

#### On client nodes (destination)
Start the iPerf3 server on the destination machine:
```bash
iperf -s -i 1
```

#### On the attacker node (source)
```bash
iperf -c 192.168.129.1 -b 4.8G -P 8 -w 1M -i 1 -t 3000 #(33%)
```
| Flag / Option      | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `-c 192.168.129.1` | Run in `client mode` and connect to the iPerf3 server (e.g., `192.168.129.1`) |
| `-b 4.8G`          | Set **target bandwidth** (e.g., 4.8) with to consume 33%/66% of the total available bandwidth |
| `-P 8`             | Use `8 parallel connections (streams)` for higher throughput            |
| `-w 1M`            | Set `TCP window size` (socket buffer) to `1 Megabyte`                 |
| `-i 1`             | Report stats `every 1 second` during the test                           |
| `-t 3000`          | Run the test for `3000 seconds` (50 minutes)                            |

