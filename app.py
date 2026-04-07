import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multiscreen Addiction Detector",
    page_icon="MS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Clean Student Project Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: #f8f9fa;
    color: #212529;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dee2e6;
}
[data-testid="stSidebar"] * {
    color: #495057 !important;
}

/* Hero title */
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    color: #1a73e8;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-size: 0.9rem;
    text-align: center;
    color: #6c757d;
    letter-spacing: 1px;
    margin-bottom: 1.5rem;
}

/* Section headers */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #1a73e8;
    border-bottom: 2px solid #1a73e8;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.metric-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.metric-label {
    font-size: 0.75rem;
    color: #6c757d;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 500;
}

/* Risk badge */
.risk-badge-severe  { background: #dc3545; }
.risk-badge-high    { background: #fd7e14; }
.risk-badge-moderate{ background: #ffc107; color:#212529 !important; }
.risk-badge-low     { background: #28a745; }
.risk-badge-minimal { background: #17a2b8; }
.risk-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    color: white;
    margin: 0.5rem 0;
}

/* Percent display */
.big-percent {
    font-family: 'Inter', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
    margin: 0.5rem 0;
}

/* Solution cards */
.sol-card {
    background: #ffffff;
    border-left: 4px solid;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.sol-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 0.4rem;
}
.sol-text {
    font-size: 0.9rem;
    color: #495057;
    line-height: 1.5;
}

/* Divider */
.neon-divider {
    height: 1px;
    background: #dee2e6;
    border: none;
    margin: 1.5rem 0;
}

/* Footer */
.footer { text-align:center; color:#adb5bd; font-size:0.75rem; margin-top:3rem; letter-spacing:1px; }

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.5px !important;
    color: #6c757d !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1a73e8 !important;
    border-bottom: 2px solid #1a73e8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODELS (cached so only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Hyper-tuning 4 AI models with GridSearch / RandomSearch CV — this may take 1-2 minutes...")
def train_models():
    import glob, os

    # Look for CSV in the app's directory (repo root for Streamlit Cloud)
    csv_files = (
        glob.glob('*.csv.xls') +
        glob.glob('data/*.csv.xls') +
        glob.glob('dataset/*.csv.xls')
    )
    if not csv_files:
        st.error(
            "Dataset CSV not found!\n\n"
            "Make sure **teen_multiscreen_addiction_dataset.csv** is committed "
            "to the root of your repository (or inside a `data/` folder)."
        )
        st.stop()

    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    df_clean = df.drop(columns=['ID', 'Name', 'Location'])

    grade_order = {'6th':6,'7th':7,'8th':8,'9th':9,'10th':10,'11th':11,'12th':12}
    df_clean['School_Grade'] = df_clean['School_Grade'].map(grade_order)

    le = LabelEncoder()
    gender_classes, purpose_classes = {}, {}
    for col in ['Gender', 'Phone_Usage_Purpose']:
        df_clean[col] = le.fit_transform(df_clean[col])
        if col == 'Gender':
            gender_classes = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
        else:
            purpose_classes = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

    # Feature engineering
    df_clean['Phone_Active_Screen']     = df_clean['Time_on_Social_Media'] + df_clean['Time_on_Gaming'] + df_clean['Time_on_Education']
    df_clean['Phone_Check_Intensity']   = df_clean['Phone_Checks_Per_Day'] / (df_clean['Daily_Usage_Hours'] + 1e-5)
    df_clean['Weekend_Weekday_Ratio']   = df_clean['Weekend_Usage_Hours'] / (df_clean['Daily_Usage_Hours'] + 1e-5)
    df_clean['Total_Laptop_Hours']      = df_clean['Laptop_Study_Hours'] + df_clean['Laptop_Gaming_TimePass_Hours'] + df_clean['Laptop_Usage_Before_Bed_Hours']
    df_clean['Laptop_Productive_Ratio'] = df_clean['Laptop_Study_Hours'] / (df_clean['Total_Laptop_Hours'] + 1e-5)
    df_clean['Gaming_Cross_Device']     = df_clean['Laptop_Gaming_TimePass_Hours'] / (df_clean['Time_on_Gaming'] + 1e-5)
    df_clean['Total_All_Screen_Hours']  = df_clean['Daily_Usage_Hours'] + df_clean['Total_Laptop_Hours']
    df_clean['Total_Before_Bed_Screen'] = df_clean['Screen_Time_Before_Bed'] + df_clean['Laptop_Usage_Before_Bed_Hours']
    df_clean['Sleep_Deficit']           = df_clean['Sleep_Hours'] - 9
    df_clean['Mental_Health_Score']     = df_clean['Anxiety_Level'] + df_clean['Depression_Level'] - df_clean['Self_Esteem']

    X = df_clean.drop(columns=['Addiction_Level'])
    y = df_clean['Addiction_Level']
    feature_cols = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

    # ── KNN Hyperparameter Tuning (GridSearchCV) ──────────────────────────
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }
    knn_search = GridSearchCV(
        KNeighborsRegressor(), knn_param_grid,
        cv=3, scoring='r2', n_jobs=-1
    )
    knn_search.fit(X_train, y_train)
    knn = knn_search.best_estimator_

    # ── SVM Hyperparameter Tuning (GridSearchCV) ─────────────────────────
    svm_param_grid = {
        'kernel': ['rbf'],
        'C': [1, 10, 50, 100],
        'epsilon': [0.01, 0.05, 0.1],
        'gamma': ['scale', 'auto'],
    }
    svm_search = GridSearchCV(
        SVR(), svm_param_grid,
        cv=3, scoring='r2', n_jobs=-1
    )
    svm_search.fit(X_train, y_train)
    svm = svm_search.best_estimator_

    # ── XGBoost Hyperparameter Tuning (RandomizedSearchCV) ───────────────
    xgb_param_dist = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(verbosity=0, random_state=42), xgb_param_dist,
        n_iter=20, cv=3, scoring='r2', n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train, y_train)
    xgb = xgb_search.best_estimator_

    # ── Random Forest Hyperparameter Tuning (RandomizedSearchCV) ─────────
    rf_param_dist = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 16, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.5, 0.7],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1), rf_param_dist,
        n_iter=20, cv=3, scoring='r2', n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_

    best_params = {
        'KNN': knn_search.best_params_,
        'SVM': svm_search.best_params_,
        'XGBoost': xgb_search.best_params_,
        'Random Forest': rf_search.best_params_,
    }

    model_scores = {
        'KNN':          round(max(0, r2_score(y_test, knn.predict(X_test))) * 100, 2),
        'SVM':          round(max(0, r2_score(y_test, svm.predict(X_test))) * 100, 2),
        'XGBoost':      round(max(0, r2_score(y_test, xgb.predict(X_test))) * 100, 2),
        'Random Forest':round(max(0, r2_score(y_test, rf.predict(X_test))) * 100, 2),
    }

    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return (knn, svm, xgb, rf, scaler, feature_cols,
            gender_classes, purpose_classes, grade_order,
            model_scores, feat_imp, df, best_params)


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTIONS BY RISK LEVEL
# ─────────────────────────────────────────────────────────────────────────────
def get_solutions(pct):
    if pct >= 85:
        color = "#dc3545"
        solutions = [
            ("IMMEDIATE DIGITAL DETOX", "#dc3545",
             "Take a 7-day complete break from all non-essential screens. Use this time to rediscover offline hobbies. Studies show even a short detox significantly resets dopamine pathways."),
            ("SEEK PROFESSIONAL HELP", "#c82333",
             "Your addiction score is critical. Consult a therapist or counselor specializing in digital addiction. Cognitive Behavioral Therapy (CBT) has 80%+ success rate for screen addiction."),
            ("ENFORCE HARD SCREEN LIMITS", "#e04050",
             "Use parental controls or apps like Screen Time / Digital Wellbeing to hard-cap daily usage at 2 hours max. Remove all gaming & social apps from your primary device immediately."),
            ("FIX YOUR SLEEP — NO SCREENS 2 HRS BEFORE BED", "#e85060",
             "Your pre-bed screen usage is critically high. Place your phone in a different room before sleeping. Replace with reading, journaling, or breathing exercises."),
            ("MANDATORY DAILY PHYSICAL ACTIVITY", "#f06070",
             "Commit to 60 mins of exercise daily — it is scientifically proven to reduce compulsive screen urges by releasing natural dopamine. Start with a daily walk or join a sport."),
            ("FAMILY INTERVENTION & ACCOUNTABILITY PARTNER", "#f08090",
             "Share your results with a trusted family member. Set up weekly check-ins. Having accountability reduces relapse by 65%."),
        ]
    elif pct >= 65:
        color = "#fd7e14"
        solutions = [
            ("APP USAGE LIMITS — STRICT SCHEDULE", "#fd7e14",
             "Set hard daily limits: Social Media <= 45 min, Gaming <= 1 hr, total phone <= 3 hrs. Use built-in Screen Time (iPhone) or Digital Wellbeing (Android) right now."),
            ("REPLACE GAMING/SOCIAL TIME WITH SKILLS", "#e8700d",
             "Convert 1 hour of gaming into learning a new skill — coding, music, drawing. Productive screen time significantly lowers addiction scores over 4-6 weeks."),
            ("NO PHONE IN BEDROOM AFTER 10 PM", "#f09030",
             "Your before-bed screen habits are harming sleep quality. Invest in a physical alarm clock, charge your phone outside the bedroom. Aim for 8+ hrs sleep."),
            ("MINDFULNESS & STRESS MANAGEMENT", "#f0a050",
             "High anxiety/depression correlates with screen overuse. Practice 10-15 min of daily meditation (Headspace, Calm). Reducing stress reduces compulsive checking."),
            ("WEEKLY SCREEN-FREE DAY", "#f0b070",
             "Pick one day per week (e.g., Sunday) as a complete screen-free day. Fill it with outdoor activities, sports, or social meetups offline."),
        ]
    elif pct >= 45:
        color = "#ffc107"
        solutions = [
            ("USE THE 20-20-20 RULE", "#ffc107",
             "Every 20 mins of screen time, look at something 20 feet away for 20 seconds. Set a timer. This breaks the compulsive scroll cycle and protects your eyes."),
            ("TURN OFF ALL NON-ESSENTIAL NOTIFICATIONS", "#e0a800",
             "Disable all social media, game, and news notifications. Research shows notifications are the #1 trigger for compulsive phone checking. Keep only calls/SMS."),
            ("SCHEDULE OUTDOOR BREAKS", "#d4a012",
             "Take a 15-minute outdoor break every 2 hours of screen use. Nature exposure for just 20 mins/day reduces screen craving significantly."),
            ("SWAP SOCIAL MEDIA FOR READING", "#c89a20",
             "Replace 30 mins of social media with reading a book or long-form article. This builds focus and gives your brain the stimulation it seeks without addiction risk."),
        ]
    elif pct >= 25:
        color = "#28a745"
        solutions = [
            ("MAINTAIN HEALTHY SCREEN HABITS", "#28a745",
             "You're doing well! Keep your phone usage balanced. Continue prioritizing sleep, exercise, and offline socializing to maintain this healthy level."),
            ("TRACK YOUR SCREEN TIME WEEKLY", "#34b553",
             "Use your phone's built-in screen time tracker to monitor weekly trends. Catching any upward creep early prevents future addiction."),
            ("SET A PERSONAL SCREEN BUDGET", "#40c561",
             "Set a daily screen budget per app category. Having an intentional limit keeps you in control even when stress increases usage temporarily."),
        ]
    else:
        color = "#17a2b8"
        solutions = [
            ("EXCELLENT SCREEN HEALTH!", "#17a2b8",
             "Your screen usage is minimal and healthy. You're in the top tier of digital wellbeing. Keep up your great habits!"),
            ("SHARE YOUR HABITS", "#1ab0c5",
             "Consider sharing your healthy screen habits with friends and family. Peer influence is powerful — you can help others reduce their addiction."),
            ("STAY CONSISTENT", "#20bdd2",
             "Continue your current routine of prioritizing offline activities, exercise, and quality sleep. Consistency is the key to lifelong digital wellness."),
        ]
    return color, solutions


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(inp_dict, knn, svm, xgb, rf, scaler, feature_cols,
            gender_classes, purpose_classes, grade_order):
    inp = inp_dict.copy()
    inp['Gender']              = gender_classes.get(inp['Gender'], 1)
    inp['School_Grade']        = grade_order.get(inp['School_Grade'], 10)
    inp['Phone_Usage_Purpose'] = purpose_classes.get(inp['Phone_Usage_Purpose'], 0)

    inp['Phone_Active_Screen']     = inp['Time_on_Social_Media'] + inp['Time_on_Gaming'] + inp['Time_on_Education']
    inp['Phone_Check_Intensity']   = inp['Phone_Checks_Per_Day'] / (inp['Daily_Usage_Hours'] + 1e-5)
    inp['Weekend_Weekday_Ratio']   = inp['Weekend_Usage_Hours'] / (inp['Daily_Usage_Hours'] + 1e-5)
    inp['Total_Laptop_Hours']      = inp['Laptop_Study_Hours'] + inp['Laptop_Gaming_TimePass_Hours'] + inp['Laptop_Usage_Before_Bed_Hours']
    inp['Laptop_Productive_Ratio'] = inp['Laptop_Study_Hours'] / (inp['Total_Laptop_Hours'] + 1e-5)
    inp['Gaming_Cross_Device']     = inp['Laptop_Gaming_TimePass_Hours'] / (inp['Time_on_Gaming'] + 1e-5)
    inp['Total_All_Screen_Hours']  = inp['Daily_Usage_Hours'] + inp['Total_Laptop_Hours']
    inp['Total_Before_Bed_Screen'] = inp['Screen_Time_Before_Bed'] + inp['Laptop_Usage_Before_Bed_Hours']
    inp['Sleep_Deficit']           = inp['Sleep_Hours'] - 9
    inp['Mental_Health_Score']     = inp['Anxiety_Level'] + inp['Depression_Level'] - inp['Self_Esteem']

    inp_df     = pd.DataFrame([inp])[feature_cols]
    inp_scaled = scaler.transform(inp_df)

    raw = {
        'KNN':           float(knn.predict(inp_scaled)[0]),
        'SVM':           float(svm.predict(inp_scaled)[0]),
        'XGBoost':       float(xgb.predict(inp_scaled)[0]),
        'Random Forest': float(rf.predict(inp_scaled)[0]),
    }
    pcts = {m: round(float(np.clip((v/10)*100, 0, 100)), 2) for m, v in raw.items()}
    ensemble = round(float(np.mean(list(pcts.values()))), 2)
    return raw, pcts, ensemble


def risk_info(pct):
    if pct >= 85:   return "SEVERE ADDICTION",  "risk-badge-severe",  "#dc3545"
    elif pct >= 65: return "HIGH RISK",          "risk-badge-high",    "#fd7e14"
    elif pct >= 45: return "MODERATE RISK",      "risk-badge-moderate","#ffc107"
    elif pct >= 25: return "LOW RISK",           "risk-badge-low",     "#28a745"
    else:           return "MINIMAL RISK",       "risk-badge-minimal", "#17a2b8"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
(knn, svm, xgb, rf, scaler, feature_cols,
 gender_classes, purpose_classes, grade_order,
 model_scores, feat_imp, df_raw, best_params) = train_models()


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">MULTISCREEN ADDICTION DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-POWERED TEEN DIGITAL WELLNESS ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

# Model accuracy pills
acc_cols = st.columns(4)
model_labels = [("KNN", "KNN"), ("SVM", "SVM"), ("XGBoost", "XGBoost"), ("Random Forest", "Random Forest")]
for col, (label, key) in zip(acc_cols, model_labels):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:#1a73e8">{model_scores[key]:.1f}%</div>
            <div class="metric-label">R² Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["PREDICT YOUR ADDICTION", "DATASET INSIGHTS", "MODEL PERFORMANCE"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Enter Your Details</div>', unsafe_allow_html=True)

    # ── SIDEBAR INPUTS ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='font-family:Inter,sans-serif;font-size:1rem;font-weight:600;
        color:#1a73e8;text-align:center;padding:0.5rem 0 1rem 0;
        border-bottom:1px solid #dee2e6;margin-bottom:1rem;'>
        INPUT PANEL
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Personal Info**")
        age          = st.slider("Age", 13, 18, 16)
        gender       = st.selectbox("Gender", ["Male", "Female", "Other"])
        grade        = st.selectbox("School Grade", ["6th","7th","8th","9th","10th","11th","12th"], index=5)

        st.markdown("---")
        st.markdown("**Phone Usage**")
        daily_usage   = st.slider("Daily Phone Usage (hrs)", 0.0, 12.0, 6.0, 0.5)
        phone_checks  = st.slider("Phone Checks Per Day", 10, 300, 80)
        apps_used     = st.slider("Apps Used Daily", 1, 30, 10)
        stb_phone     = st.slider("Phone Screen Before Bed (hrs)", 0.0, 3.0, 1.0, 0.1)
        weekend_usage = st.slider("Weekend Phone Usage (hrs)", 0.0, 15.0, 6.0, 0.5)
        purpose       = st.selectbox("Primary Phone Purpose", ["Browsing","Gaming","Education","Social Media","Other"])

        st.markdown("**Phone Time Breakdown**")
        t_social  = st.slider("Social Media (hrs)", 0.0, 6.0, 1.5, 0.1)
        t_gaming  = st.slider("Gaming on Phone (hrs)", 0.0, 5.0, 1.0, 0.1)
        t_edu     = st.slider("Education on Phone (hrs)", 0.0, 4.0, 0.8, 0.1)

        st.markdown("---")
        st.markdown("**Laptop Usage**")
        lap_study  = st.slider("Laptop Study (hrs/day)", 0.0, 6.0, 2.0, 0.1)
        lap_gaming = st.slider("Laptop Gaming/Timepass (hrs/day)", 0.0, 6.0, 1.0, 0.1)
        lap_bed    = st.slider("Laptop Before Bed (hrs)", 0.0, 3.0, 0.8, 0.1)

        st.markdown("---")
        st.markdown("**Health & Lifestyle**")
        sleep    = st.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.5)
        exercise = st.slider("Exercise (hrs/day)", 0.0, 4.0, 0.5, 0.1)
        academic = st.slider("Academic Performance (50-100)", 50, 100, 72)
        social_i = st.slider("Social Interactions (1-10)", 1, 10, 5)
        family_c = st.slider("Family Communication (1-10)", 1, 10, 5)

        st.markdown("**Mental Health**")
        anxiety    = st.slider("Anxiety Level (1-10)", 1, 10, 5)
        depression = st.slider("Depression Level (1-10)", 1, 10, 4)
        self_est   = st.slider("Self Esteem (1-10)", 1, 10, 6)

        st.markdown("**Parental Control**")
        parental = st.radio("Parental Control Active?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    # ── COLLECT INPUT ─────────────────────────────────────────────────────────
    user_input = {
        'Age': age, 'Gender': gender, 'School_Grade': grade,
        'Daily_Usage_Hours': daily_usage, 'Sleep_Hours': sleep,
        'Academic_Performance': academic, 'Social_Interactions': social_i,
        'Exercise_Hours': exercise, 'Anxiety_Level': anxiety,
        'Depression_Level': depression, 'Self_Esteem': self_est,
        'Parental_Control': parental, 'Screen_Time_Before_Bed': stb_phone,
        'Phone_Checks_Per_Day': phone_checks, 'Apps_Used_Daily': apps_used,
        'Time_on_Social_Media': t_social, 'Time_on_Gaming': t_gaming,
        'Time_on_Education': t_edu, 'Phone_Usage_Purpose': purpose,
        'Family_Communication': family_c, 'Weekend_Usage_Hours': weekend_usage,
        'Laptop_Study_Hours': lap_study,
        'Laptop_Gaming_TimePass_Hours': lap_gaming,
        'Laptop_Usage_Before_Bed_Hours': lap_bed,
    }

    # Auto-predict on load / on change
    raw_preds, pct_preds, ensemble_pct = predict(
        user_input, knn, svm, xgb, rf, scaler, feature_cols,
        gender_classes, purpose_classes, grade_order
    )
    risk_text, risk_class, risk_color = risk_info(ensemble_pct)
    risk_color_sol, solutions = get_solutions(ensemble_pct)

    # ── RESULT DISPLAY ────────────────────────────────────────────────────────
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown(f"""
        <div style='text-align:center; padding:1.5rem 0;'>
            <div style='color:#6c757d;font-family:Inter,sans-serif;font-size:0.8rem;
            letter-spacing:2px;font-weight:500;'>ENSEMBLE ADDICTION SCORE</div>
            <div class="big-percent" style='color:{risk_color};'>
                {ensemble_pct:.1f}%
            </div>
            <div class='risk-badge {risk_class}'>{risk_text}</div>
        </div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ensemble_pct,
            number={'suffix': '%', 'font': {'size': 40, 'color': risk_color, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'color': '#6c757d'}, 'tickcolor': '#6c757d'},
                'bar':  {'color': risk_color, 'thickness': 0.25},
                'bgcolor': '#f1f3f5',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 25],  'color': '#e3f2fd'},
                    {'range': [25, 45], 'color': '#e8f5e9'},
                    {'range': [45, 65], 'color': '#fff8e1'},
                    {'range': [65, 85], 'color': '#fff3e0'},
                    {'range': [85, 100],'color': '#fce4ec'},
                ],
                'threshold': {'line': {'color': '#212529', 'width': 3},
                              'thickness': 0.8, 'value': ensemble_pct}
            },
            title={'text': "ADDICTION LEVEL", 'font': {'size': 14, 'color': '#6c757d', 'family': 'Inter'}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#212529', height=280, margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, width='stretch')

        model_names = list(pct_preds.keys())
        model_vals  = list(pct_preds.values())
        bar_clrs    = [risk_color if v >= 85 else '#fd7e14' if v >= 65
                       else '#ffc107' if v >= 45 else '#28a745' for v in model_vals]

        fig_bar = go.Figure(go.Bar(
            x=model_names, y=model_vals,
            marker=dict(color=bar_clrs, line=dict(color='rgba(0,0,0,0.1)', width=1)),
            text=[f'{v:.1f}%' for v in model_vals], textposition='outside',
            textfont=dict(family='Inter', size=13, color='#212529')
        ))
        fig_bar.add_hline(y=ensemble_pct, line_dash="dash", line_color="#495057",
                          annotation_text=f"Ensemble: {ensemble_pct:.1f}%",
                          annotation_font_color="#495057")
        fig_bar.update_layout(
            title=dict(text="PREDICTION BY EACH MODEL", font=dict(family='Inter', size=12, color='#6c757d')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa',
            font_color='#212529', height=280,
            yaxis=dict(range=[0, 110], gridcolor='#dee2e6', title='Addiction %'),
            xaxis=dict(gridcolor='#dee2e6'),
            margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_bar, width='stretch')

    with col_side:
        categories = ['Phone\nUsage', 'Social\nMedia', 'Gaming', 'Laptop\nTotal', 'Before\nBed']
        max_vals   = [12, 6, 5, 12, 5]
        user_vals  = [daily_usage, t_social, t_gaming + lap_gaming,
                      lap_study + lap_gaming + lap_bed, stb_phone + lap_bed]
        norm_vals  = [min(v/m, 1)*10 for v, m in zip(user_vals, max_vals)]

        fig_radar = go.Figure(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba({int(risk_color[1:3],16)},{int(risk_color[3:5],16)},{int(risk_color[5:7],16)},0.2)',
            line=dict(color=risk_color, width=2),
            marker=dict(color=risk_color, size=6)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#f8f9fa',
                radialaxis=dict(visible=True, range=[0, 10],
                                gridcolor='#dee2e6', tickfont=dict(color='#adb5bd')),
                angularaxis=dict(gridcolor='#dee2e6', tickfont=dict(color='#495057', size=11))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(text="SCREEN TIME PROFILE", font=dict(family='Inter', size=11, color='#6c757d')),
            height=280, margin=dict(t=50, b=10, l=40, r=40),
            showlegend=False
        )
        st.plotly_chart(fig_radar, width='stretch')

        for model, pct in pct_preds.items():
            clr = risk_info(pct)[2]
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #dee2e6;
            border-left:4px solid {clr};
            border-radius:4px;padding:0.6rem 1rem;margin:0.4rem 0;
            display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-family:Inter,sans-serif;font-size:0.8rem;
                color:#6c757d;font-weight:500;'>{model}</span>
                <span style='font-family:Inter,sans-serif;font-size:1.1rem;
                font-weight:700;color:{clr};'>{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

    # ── SOLUTIONS ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="section-header" style='color:{risk_color_sol};'>
        PERSONALIZED SOLUTIONS — {risk_text}
    </div>
    """, unsafe_allow_html=True)

    sol_cols = st.columns(2)
    for i, (title, sol_color, text) in enumerate(solutions):
        with sol_cols[i % 2]:
            st.markdown(f"""
            <div class="sol-card" style='border-color:{sol_color};'>
                <div class="sol-title" style='color:{sol_color};'>{title}</div>
                <div class="sol-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── SCREEN TIME BREAKDOWN PIE ─────────────────────────────────────────────
    st.markdown('<div class="section-header">YOUR SCREEN TIME BREAKDOWN</div>', unsafe_allow_html=True)

    pie_labels = ['Social Media (Phone)', 'Gaming (Phone)', 'Education (Phone)',
                  'Laptop Study', 'Laptop Gaming', 'Laptop Before Bed', 'Other Phone Time']
    other_phone = max(0, daily_usage - t_social - t_gaming - t_edu)
    pie_values  = [t_social, t_gaming, t_edu, lap_study, lap_gaming, lap_bed, other_phone]
    pie_colors  = ['#1a73e8', '#dc3545', '#17a2b8', '#28a745', '#fd7e14', '#6f42c1', '#ffc107']
    pie_values_clean = [max(0, v) for v in pie_values]

    if sum(pie_values_clean) > 0:
        fig_pie = go.Figure(go.Pie(
            labels=pie_labels, values=pie_values_clean, hole=0.5,
            marker=dict(colors=pie_colors, line=dict(color='#ffffff', width=2)),
            textfont=dict(family='Inter', size=12),
            hovertemplate='<b>%{label}</b><br>%{value:.1f} hrs/day<br>%{percent}<extra></extra>'
        ))
        fig_pie.add_annotation(
            text=f"{sum(pie_values_clean):.1f} hrs<br>total/day",
            x=0.5, y=0.5, font=dict(family='Inter', size=13, color='#212529'),
            showarrow=False
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font_color='#212529',
            height=350, margin=dict(t=20, b=20, l=0, r=0),
            legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_pie, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    for col, (val, label, color) in zip(
        [kpi1, kpi2, kpi3, kpi4],
        [(len(df_raw), "TOTAL RECORDS", "#1a73e8"),
         (len(df_raw.columns), "FEATURES", "#6f42c1"),
         (f"{df_raw['Addiction_Level'].mean():.1f}/10", "AVG ADDICTION", "#dc3545"),
         (f"{df_raw['Daily_Usage_Hours'].mean():.1f}h",  "AVG DAILY USE", "#fd7e14")]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style='color:{color};font-size:1.8rem;'>{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(df_raw, x='Addiction_Level', nbins=20,
                                 color_discrete_sequence=['#1a73e8'],
                                 title='Addiction Level Distribution')
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='#f8f9fa',
                                font_color='#212529', height=300,
                                title_font=dict(family='Inter', size=12, color='#6c757d'))
        fig_hist.update_xaxes(gridcolor='#dee2e6')
        fig_hist.update_yaxes(gridcolor='#dee2e6')
        st.plotly_chart(fig_hist, width='stretch')

    with c2:
        fig_box = px.box(df_raw,
                          x='Phone_Usage_Purpose', y='Addiction_Level',
                          color='Phone_Usage_Purpose',
                          color_discrete_sequence=['#1a73e8','#6f42c1','#17a2b8','#28a745','#fd7e14'],
                          title='Addiction by Phone Purpose')
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='#f8f9fa',
                               font_color='#212529', height=300, showlegend=False,
                               title_font=dict(family='Inter', size=12, color='#6c757d'))
        fig_box.update_xaxes(gridcolor='#dee2e6')
        fig_box.update_yaxes(gridcolor='#dee2e6')
        st.plotly_chart(fig_box, width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        fig_sc1 = px.scatter(df_raw, x='Daily_Usage_Hours', y='Addiction_Level',
                              color='Addiction_Level', color_continuous_scale='blues',
                              opacity=0.5, title='Daily Usage vs Addiction')
        fig_sc1.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='#f8f9fa',
                               font_color='#212529', height=300,
                               title_font=dict(family='Inter', size=12, color='#6c757d'))
        fig_sc1.update_xaxes(gridcolor='#dee2e6')
        fig_sc1.update_yaxes(gridcolor='#dee2e6')
        st.plotly_chart(fig_sc1, width='stretch')

    with c4:
        fig_sc2 = px.scatter(df_raw, x='Sleep_Hours', y='Addiction_Level',
                              color='Anxiety_Level', color_continuous_scale='reds',
                              opacity=0.5, title='Sleep Hours vs Addiction (colored by Anxiety)')
        fig_sc2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='#f8f9fa',
                               font_color='#212529', height=300,
                               title_font=dict(family='Inter', size=12, color='#6c757d'))
        fig_sc2.update_xaxes(gridcolor='#dee2e6')
        fig_sc2.update_yaxes(gridcolor='#dee2e6')
        st.plotly_chart(fig_sc2, width='stretch')

    st.markdown('<div class="section-header">Laptop Usage Analysis</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    laptop_cols_plot = [
        ('Laptop_Study_Hours', '#28a745', 'Laptop Study vs Addiction'),
        ('Laptop_Gaming_TimePass_Hours', '#dc3545', 'Laptop Gaming vs Addiction'),
        ('Laptop_Usage_Before_Bed_Hours', '#6f42c1', 'Laptop Before Bed vs Addiction'),
    ]
    for col, (xcol, clr, title) in zip([lc1, lc2, lc3], laptop_cols_plot):
        with col:
            fig_l = px.scatter(df_raw, x=xcol, y='Addiction_Level', opacity=0.4,
                                color_discrete_sequence=[clr], title=title)
            fig_l.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                 plot_bgcolor='#f8f9fa',
                                 font_color='#212529', height=280,
                                 title_font=dict(family='Inter', size=10, color='#6c757d'),
                                 margin=dict(t=40,b=20,l=20,r=20))
            fig_l.update_xaxes(gridcolor='#dee2e6')
            fig_l.update_yaxes(gridcolor='#dee2e6')
            st.plotly_chart(fig_l, width='stretch')

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        gender_avg = df_raw.groupby('Gender')['Addiction_Level'].mean().reset_index()
        fig_g = px.bar(gender_avg, x='Gender', y='Addiction_Level',
                        color='Gender', color_discrete_sequence=['#1a73e8','#6f42c1','#17a2b8'],
                        title='Avg Addiction by Gender')
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa',
                             font_color='#212529', height=280, showlegend=False,
                             title_font=dict(family='Inter', size=12, color='#6c757d'))
        fig_g.update_xaxes(gridcolor='#dee2e6'); fig_g.update_yaxes(gridcolor='#dee2e6')
        st.plotly_chart(fig_g, width='stretch')

    with d2:
        grade_avg = df_raw.groupby('School_Grade')['Addiction_Level'].mean().reset_index()
        grade_order_list = ['6th','7th','8th','9th','10th','11th','12th']
        grade_avg['School_Grade'] = pd.Categorical(grade_avg['School_Grade'],
                                                    categories=grade_order_list, ordered=True)
        grade_avg = grade_avg.sort_values('School_Grade')
        fig_gr = px.line(grade_avg, x='School_Grade', y='Addiction_Level',
                          markers=True, title='Avg Addiction by School Grade',
                          color_discrete_sequence=['#1a73e8'])
        fig_gr.update_traces(line=dict(width=3), marker=dict(size=10, color='#dc3545'))
        fig_gr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa',
                              font_color='#212529', height=280,
                              title_font=dict(family='Inter', size=12, color='#6c757d'))
        fig_gr.update_xaxes(gridcolor='#dee2e6'); fig_gr.update_yaxes(gridcolor='#dee2e6')
        st.plotly_chart(fig_gr, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Accuracy Comparison</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, (model, score) in zip([mc1, mc2, mc3, mc4], model_scores.items()):
        clr = "#28a745" if score >= 70 else "#ffc107" if score >= 50 else "#fd7e14"
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{model}</div>
                <div class="metric-value" style='color:{clr};'>{score:.1f}%</div>
                <div class="metric-label">R² Score</div>
            </div>
            """, unsafe_allow_html=True)

    fig_acc = go.Figure(go.Bar(
        x=list(model_scores.keys()),
        y=list(model_scores.values()),
        marker=dict(
            color=['#1a73e8','#fd7e14','#dc3545','#28a745'],
            line=dict(color='rgba(0,0,0,0.1)', width=1)
        ),
        text=[f'{v:.1f}%' for v in model_scores.values()],
        textposition='outside', textfont=dict(family='Inter', size=14, color='#212529')
    ))
    fig_acc.update_layout(
        title=dict(text='MODEL R² ACCURACY (%)', font=dict(family='Inter', size=13, color='#6c757d')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa',
        font_color='#212529', height=350,
        yaxis=dict(range=[0, 110], gridcolor='#dee2e6', title='Accuracy %'),
        xaxis=dict(gridcolor='#dee2e6'),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_acc, width='stretch')

    # ── Best Hyperparameters Found ────────────────────────────────────────
    st.markdown('<div class="section-header">Best Hyperparameters (GridSearch / RandomSearch CV)</div>',
                unsafe_allow_html=True)

    bp_cols = st.columns(2)
    for i, (model_name, params) in enumerate(best_params.items()):
        with bp_cols[i % 2]:
            params_html = "".join([
                f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;"
                f"border-bottom:1px solid #dee2e6;'>"
                f"<span style='color:#6c757d;font-size:0.85rem;'>{k}</span>"
                f"<span style='color:#1a73e8;font-family:Inter,sans-serif;font-size:0.85rem;font-weight:600;'>{v}</span>"
                f"</div>"
                for k, v in params.items()
            ])
            st.markdown(f"""
            <div class="metric-card" style='text-align:left;'>
                <div class="metric-label" style='text-align:center;margin-bottom:0.8rem;
                font-size:0.9rem;color:#1a73e8;font-weight:600;'>{model_name}</div>
                {params_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    top_n = 20
    fi_df = feat_imp.head(top_n).reset_index()
    fi_df.columns = ['Feature', 'Importance']
    fi_colors = ['#dc3545' if i < 5 else '#fd7e14' if i < 10 else '#1a73e8'
                 for i in range(len(fi_df))]

    fig_fi = go.Figure(go.Bar(
        x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
        marker=dict(color=fi_colors, line=dict(color='rgba(0,0,0,0.05)', width=1)),
        text=[f'{v:.4f}' for v in fi_df['Importance']],
        textposition='outside', textfont=dict(size=10, color='#212529')
    ))
    fig_fi.update_layout(
        title=dict(text=f'TOP {top_n} FEATURES  (Red=Top5  Orange=Top10  Blue=Rest)',
                   font=dict(family='Inter', size=12, color='#6c757d')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f8f9fa',
        font_color='#212529', height=600,
        yaxis=dict(autorange='reversed', gridcolor='#dee2e6', tickfont=dict(size=11)),
        xaxis=dict(gridcolor='#dee2e6', title='Importance Score'),
        margin=dict(t=50, b=20, l=200, r=80)
    )
    st.plotly_chart(fig_fi, width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    MULTISCREEN ADDICTION DETECTOR  |  POWERED BY KNN | SVM | XGBOOST | RANDOM FOREST
    <br>TRAINED ON 3000 TEEN RECORDS  |  28 FEATURES  |  AI-POWERED DIGITAL WELLNESS ANALYSIS
</div>
""", unsafe_allow_html=True)
