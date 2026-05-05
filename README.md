# Power-Analysis-in-side-channel-attack
Power-SCA: Side-Channel Power Analysis Simulator A Python-based simulation environment for demonstrating and analyzing Simple Power Analysis (SPA) and Differential Power Analysis (DPA). This project provides a visual bridge between cryptographic theory and hardware leakage realities, featuring a web-based interface for interactive experimentation.
This is a comprehensive GitHub `README.md` template for a project that simulates Power Analysis (Side-Channel Attacks). It uses a clean, professional structure suitable for portfolios or academic submissions.

---

# 🛡️ Power-SCA: Side-Channel Power Analysis Simulator

A Python-based simulation environment for demonstrating and analyzing **Simple Power Analysis (SPA)** and **Differential Power Analysis (DPA)**. This project provides a visual bridge between cryptographic theory and hardware leakage realities, featuring a web-based interface for interactive experimentation.

---

## 📖 Overview
Side-channel attacks (SCA) exploit physical information leaked by hardware during cryptographic operations. This project simulates the power consumption of a device executing an encryption algorithm (e.g., AES-128) and provides tools to recover secret keys using statistical analysis.

### Key Features
*   **Leakage Modeling:** Implements **Hamming Weight** and **Hamming Distance** power models.
*   **Visual Analysis:** Real-time power trace plotting using **Matplotlib**.
*   **Statistical Attacks:** Support for Correlation Power Analysis (CPA) to isolate subkeys from noise.
*   **Web Dashboard:** An **HTML/CSS/JS** interface to adjust simulation parameters (noise levels, trace counts, etc.) without touching the code.

---

##  Technologies Used
| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3.x | Core logic, AES simulation, and statistical analysis. |
| **Data Viz** | Matplotlib / Seaborn | Generating power trace graphs and correlation heatmaps. |
| **Frontend** | HTML5, CSS3, JS | Interactive dashboard for simulation control. |
| **API/Web** | Flask / FastAPI | Connecting the Python simulation engine to the web UI. |
| **Math** | NumPy | High-performance trace manipulation and correlation. |

---

## 🛠️ System Architecture
The project is divided into three primary modules:
1.  **The Target:** A simulated cryptographic core that "leaks" power traces based on secret keys and input data.
2.  **The Analyzer:** A statistical engine that performs Pearson correlation on collected traces to identify the most likely key candidates.
3.  **The Interface:** A responsive web UI to visualize the "Ghost in the Machine."



---

##  Simulation Models
To make the simulation realistic, the project accounts for:
*   **Signal-to-Noise Ratio (SNR):** Users can inject Gaussian noise to see how many more traces are required for a successful attack.
*   **The Ghost Peak:** Visualizing the exact moment in the power trace where the S-Box substitution occurs.

### Differential Power Analysis (DPA)
By averaging traces where a specific bit is '0' vs '1', we can see a clear "differential" spike that reveals the key bit.



---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/power-sca-sim.git
cd power-sca-sim
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Simulation
```bash
python app.py
```
*Navigate to `http://localhost:5000` to access the web interface.*

---

##  Example Results
| Attack Type | Traces Required | Accuracy |
| :--- | :--- | :--- |
| Simple Power Analysis (SPA) | 1-5 | Low (Requires visual inspection) |
| Correlation Power Analysis (CPA) | 50 - 500 | High (Automated) |

---

Contributing
Contributions are welcome! If you'd like to add support for **RSA** or **Electromagnetic (EM)** leakage models, please open an issue or submit a pull request.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.

Disclaimer: This project is for educational purposes only. It is intended to help developers understand hardware vulnerabilities to build more secure, side-channel-resistant implementations.
