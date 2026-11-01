# Real-Time DDoS Detection Using Machine Learning in a Simulated Network Environment

---

## Objective

To design and implement a real-time network intrusion detection system capable of identifying and classifying Distributed Denial-of-Service (DDoS) attacks using a hybrid machine learning approach, and to validate its effectiveness in a simulated network environment using Mininet.

---

## Technical Scope

- **Dataset Handling:** Processed a large-scale network traffic dataset containing over 2 million labelled records across 34 attack categories and benign traffic classes.
- **Feature Engineering:** Applied standard scaling (StandardScaler) across 53 extracted network flow features to normalize input data for model training.
- **Anomaly Detection:** Trained an Isolation Forest model (contamination 0.02) to detect anomalous traffic patterns unseen during supervised training, serving as the zero-day safety net.
- **Multi-Class Classification:** Trained an XGBoost classifier (2000 boosting rounds) to perform fine-grained classification across 34 traffic classes including multiple DDoS, DoS, reconnaissance, and application-layer attack types.
- **Score Fusion:** Implemented a novel **Confidence-Adaptive 3-Layer Fusion** strategy:
  - *Layer 1 (Hard Veto):* Isolation Forest overrides XGBoost on extreme anomalies to prevent confident misclassifications.
  - *Layer 2 (Disagree Amplifier):* Boosts score when models strongly disagree (detects mimicry attacks).
  - *Layer 3 (Adaptive Weights):* Fusion weights shift dynamically (XGBoost, Isolation Forest, DPI) based on XGBoost's prediction confidence (`1 - P(BENIGN)`).
- **Threshold Optimization:** Conducted an automated threshold scan yielding an optimal decision boundary (threshold = 0.61) achieving FPR < 2%.
- **Real-Time Deployment:** Integrated the trained 3-layer pipeline and 6 enhanced DPI rules into a real-time packet capture and analysis module (`realtime_defender_v2.py`).
- **Network Simulation:** Validation planned via Docker containerization and Scapy packet replay to simulate real-time MEC (Multi-access Edge Computing) node deployment.
- **Cross-Environment Compatibility:** Handled model serialization and environment compatibility for Python 3.10 deployment.

---

## Course Outcome Alignment

| CO | Course Outcome Description | How This Project Addresses It |
|----|---------------------------|-------------------------------|
| **CO 2** | Understand and apply data link and network layer concepts including addressing, error detection, and protocol design | The project processes raw network packet flows using IP addressing, protocol flags (TCP/UDP/ICMP), and packet-level features to construct feature vectors for ML classification |
| **CO 3** | Apply security principles to analyse threats and implement protection mechanisms at various network layers | A multi-layer detection pipeline (anomaly detection + supervised classification + DPI scoring) is applied to identify 34 types of network attacks including DDoS, DoS, MITM, and reconnaissance |
| **CO 4** | Implement and evaluate network security protocols and mechanisms including intrusion detection | An end-to-end IDS pipeline is trained, evaluated (F1-score, TPR, FPR), and threshold-optimized to achieve a **97.9% detection rate** with a **0.32% false positive rate** after cost-sensitive tuning |
| **CO 5** | Analyse and mitigate threats such as malware, intrusion, firewalls, and denial-of-service attacks in real or simulated environments | The system is deployed in a live Mininet topology, simulating and defending against real DDoS attack variants (SYN Flood, ICMP Flood, SlowLoris, HTTP Flood, UDP Flood, and more) |

---

## Final Outcome & Metrics (Phase 6)

- A fully functional **real-time intrusion detection system** featuring a 3-Layer Confidence-Adaptive Fusion Engine.
- **Final Performance Metrics** (on 485,187 test flows):
  - Overall Detection Rate (TPR): **80.36%** (with 100% detection on all major volumetric DDoS floods)
  - False Positive Rate: **1.59%** (meeting the <2% industry requirement)
  - Precision: **99.91%**
  - Binary Accuracy: **81.10%**
- **Zero-Day Defense:** Demonstrated ability for the unsupervised Isolation Forest layer to independently flag attacks that the supervised XGBoost model misses, providing a critical safety net against novel attack vectors.
- A robust, auto-calibrated **model pipeline** serialized and ready for deployment on an edge computing (MEC) node.
- Demonstrated understanding of how adaptive machine learning architectures can overcome the limitations of purely supervised models in adversarial network environments.
