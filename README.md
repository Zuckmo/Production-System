# 🏭 Integrated Production Analytics System
### Predictive Maintenance & Quality Control for Manufacturing

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12+-blue)

---

##  Overview
Integrated system for monitoring and optimizing manufacturing operations using machine learning and analytics.

**Goals:**
- Reduce machine downtime  
- Predict product defects  
- Improve production efficiency (OEE)  

---

##  Problem
Manufacturing challenges:
- Unplanned machine failures  
- Scattered data across systems  
- Late defect detection  
- Difficult bottleneck identification  
- Inefficient maintenance strategy  

**Impact:** High cost, low efficiency, wasted resources.

---

##  Features

### 🔹 OEE Monitoring
- Tracks Availability, Performance, Quality  
- Benchmark comparison (85%)

### 🔹 Machine Condition Detection
- K-Means clustering  
- States: Healthy, Degrading, Critical  

### 🔹 Defect Visualization
- Random Forest 
- Based on sensor data  

### 🔹 Production Analysis
- Identify bottlenecks  
- Compare machine & shift performance  

### 🔹 Interactive Dashboard
- Built with Streamlit  
- Real-time monitoring  

---

##  Architecture

- **Data:** Machines, sensors, ERP
- **Processing:** Cleaning, scaling, feature engineering  
- **Models:** K-Means, Random Forest
- **Output:** Streamlit dashboard  

---

##  Tech Stack

- Python  
- Pandas, NumPy, Matplotlib  
- Scikit-learn, KMeans, RandomForest
- Streamlit  
 

---

##  Installation

git clone https://github.com/Zuckmo/Production-System.git
cd Production-System

python -m venv venv

### Windows
venv\Scripts\activate

### macOS/Linux
source venv/bin/activate

---
## check dependencies

pip install -r requirements.txt

---
## Run Streamlit

streamlit run Stream_production_app.py

---
## project Structure
Production-System/

├── Data/              # Dataset
├── Scripts/           # Python scripts
├── Models/            # Trained models
├── Reports/           # Analysis results
├── Documentation/     # Docs
└── README.md

---
## Key Insights
- OEE is low (~16%) → high improvement potential
- Machines grouped into 3 condition states
- Temperature affects efficiency and defects
- Certain products dominate defect rates
- Shift performance varies significantly


## Author
Guruh Sukmo





