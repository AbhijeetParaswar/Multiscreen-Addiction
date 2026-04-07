# 📱💻 Multiscreen Addiction Detector

> **AI-Powered Teen Digital Wellness Analyzer** — built with Streamlit, trained on 3,000 teen records using KNN, SVM, XGBoost & Random Forest.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📋 Features

- **4 ML Models** — KNN, SVM, XGBoost, Random Forest with ensemble prediction
- **Real-time Prediction** — Sliders update the addiction score instantly
- **Personalized Solutions** — Risk-level-specific wellness recommendations
- **Dataset Insights** — Interactive charts exploring the full dataset
- **Model Performance** — R² accuracy scores and feature importance plots
- **Dark Neon UI** — Custom CSS theme with animated gradients

---

## 🗂️ Repository Structure

```
your-repo/
│
├── app.py                              ← Main Streamlit app
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
└── teen_multiscreen_addiction_dataset.csv  ← Dataset (REQUIRED)
```

> ⚠️ **The CSV file must be present in the repo root.** The app will not start without it.

---

## ⚡ Deploy to Streamlit Cloud (Step-by-Step)

### 1. Prepare your GitHub repo

```bash
# Create a new repo (or use existing)
git init
git add app.py requirements.txt README.md teen_multiscreen_addiction_dataset.csv
git commit -m "Initial commit — Multiscreen Addiction Detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Select your **repository**, **branch** (`main`), and set **Main file path** to `app.py`
4. Click **"Deploy!"**

Your app will be live at:
```
https://YOUR_USERNAME-YOUR_REPO-app-XXXX.streamlit.app
```

---

## 🧪 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📊 Dataset

The app expects a CSV named `teen_multiscreen_addiction_dataset.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `ID`, `Name`, `Location` | Dropped before training |
| `Age`, `Gender`, `School_Grade` | Demographics |
| `Daily_Usage_Hours`, `Weekend_Usage_Hours` | Phone usage |
| `Phone_Checks_Per_Day`, `Apps_Used_Daily` | Behavior patterns |
| `Time_on_Social_Media`, `Time_on_Gaming`, `Time_on_Education` | Phone time breakdown |
| `Screen_Time_Before_Bed` | Pre-sleep screen use |
| `Phone_Usage_Purpose` | Primary use category |
| `Laptop_Study_Hours`, `Laptop_Gaming_TimePass_Hours`, `Laptop_Usage_Before_Bed_Hours` | Laptop usage |
| `Sleep_Hours`, `Exercise_Hours` | Health metrics |
| `Anxiety_Level`, `Depression_Level`, `Self_Esteem` | Mental health (1–10) |
| `Academic_Performance` | School score (50–100) |
| `Social_Interactions`, `Family_Communication` | Social metrics (1–10) |
| `Parental_Control` | Binary (0/1) |
| `Addiction_Level` | **Target variable** (0–10) |

---

## 🤖 Models & Feature Engineering

10 engineered features are created from raw inputs before training:

| Engineered Feature | Formula |
|---|---|
| `Phone_Active_Screen` | Social + Gaming + Education hours |
| `Phone_Check_Intensity` | Checks per day / Daily usage |
| `Weekend_Weekday_Ratio` | Weekend usage / Daily usage |
| `Total_Laptop_Hours` | Study + Gaming + Bed laptop hours |
| `Laptop_Productive_Ratio` | Study hours / Total laptop hours |
| `Gaming_Cross_Device` | Laptop gaming / Phone gaming |
| `Total_All_Screen_Hours` | Phone + Laptop total |
| `Total_Before_Bed_Screen` | Phone + Laptop before bed |
| `Sleep_Deficit` | Sleep hours − 9 |
| `Mental_Health_Score` | Anxiety + Depression − Self Esteem |

---

## 🎨 Risk Level Legend

| Score | Level | Color |
|-------|-------|-------|
| 85–100% | 🔴 Severe Addiction | Red |
| 65–84%  | 🟠 High Risk | Orange |
| 45–64%  | 🟡 Moderate Risk | Yellow |
| 25–44%  | 🟢 Low Risk | Green |
| 0–24%   | 🔵 Minimal Risk | Cyan |

---

## 🛠️ Tech Stack

- **Frontend** — Streamlit + Custom CSS (Dark Neon Theme)
- **Visualization** — Plotly (gauge, bar, radar, pie, scatter, histogram)
- **ML** — scikit-learn (KNN, SVM, Random Forest) + XGBoost
- **Data** — pandas, numpy

---

## 📄 License

MIT License — free to use and modify.
