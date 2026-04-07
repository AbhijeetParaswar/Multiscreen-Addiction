import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Neon Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #1a0a2e 100%);
    color: #e0e0ff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d2b 0%, #1a0a2e 100%);
    border-right: 1px solid #7b2fff44;
}
[data-testid="stSidebar"] * {
    color: #c8b4ff !important;
}

/* Hero title */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #ff6ec7, #7b2fff, #00d4ff, #ff6ec7);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s ease infinite;
    letter-spacing: 2px;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-family: 'Orbitron', monospace;
    font-size: 0.95rem;
    text-align: center;
    color: #8888bb;
    letter-spacing: 4px;
    margin-bottom: 2rem;
}
@keyframes gradientShift {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}

/* Section headers */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 3px;
    border-bottom: 1px solid #00d4ff44;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem 0;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a3e 0%, #2a1a4e 100%);
    border: 1px solid #7b2fff66;
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 0 20px #7b2fff22;
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 1rem;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 30px #7b2fff55;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    margin: 0.3rem 0;
}
.metric-label {
    font-size: 0.8rem;
    color: #8888bb;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Risk badge */
.risk-badge-severe  { background: linear-gradient(135deg,#ff1744,#ff5252); }
.risk-badge-high    { background: linear-gradient(135deg,#ff6d00,#ffab40); }
.risk-badge-moderate{ background: linear-gradient(135deg,#ffd600,#ffe57f); color:#111 !important; }
.risk-badge-low     { background: linear-gradient(135deg,#00c853,#69f0ae); color:#111 !important; }
.risk-badge-minimal { background: linear-gradient(135deg,#00bcd4,#80deea); color:#111 !important; }
.risk-badge {
    display: inline-block;
    padding: 0.6rem 1.8rem;
    border-radius: 50px;
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    margin: 0.5rem 0;
}

/* Percent display */
.big-percent {
    font-family: 'Orbitron', monospace;
    font-size: 5rem;
    font-weight: 900;
    text-align: center;
    line-height: 1;
    margin: 0.5rem 0;
}

/* Solution cards */
.sol-card {
    background: linear-gradient(135deg, #0f1f3d 0%, #1a2a50 100%);
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.sol-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.sol-text {
    font-size: 0.95rem;
    color: #c8c8e8;
    line-height: 1.5;
}

/* Input sliders & widgets */
.stSlider > div > div > div > div {
    background: #7b2fff !important;
}
.stSelectbox > div, .stNumberInput > div {
    background: #1a1a3e !important;
    border: 1px solid #7b2fff66 !important;
    border-radius: 8px !important;
    color: #e0e0ff !important;
}

/* Divider */
.neon-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #7b2fff, #00d4ff, #7b2fff, transparent);
    border: none;
    margin: 1.5rem 0;
}

/* Pulse animation for severe */
@keyframes pulse {
    0%  { box-shadow: 0 0 0 0 rgba(255,23,68,0.7); }
    70% { box-shadow: 0 0 0 15px rgba(255,23,68,0); }
    100%{ box-shadow: 0 0 0 0 rgba(255,23,68,0); }
}
.pulse { animation: pulse 2s infinite; }

/* Footer */
.footer { text-align:center; color:#555577; font-size:0.75rem; margin-top:3rem; letter-spacing:2px; }

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    color: #8888bb !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODELS (cached so only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training AI models on 3000 teens dataset...")
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
            "❌ Dataset CSV not found!\n\n"
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

    knn = KNeighborsRegressor(n_neighbors=7, metric='euclidean', weights='distance')
    knn.fit(X_train, y_train)

    svm = SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')
    svm.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=5,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    model_scores = {
        'KNN':          round(max(0, r2_score(y_test, knn.predict(X_test))) * 100, 2),
        'SVM':          round(max(0, r2_score(y_test, svm.predict(X_test))) * 100, 2),
        'XGBoost':      round(max(0, r2_score(y_test, xgb.predict(X_test))) * 100, 2),
        'Random Forest':round(max(0, r2_score(y_test, rf.predict(X_test))) * 100, 2),
    }

    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return (knn, svm, xgb, rf, scaler, feature_cols,
            gender_classes, purpose_classes, grade_order,
            model_scores, feat_imp, df)


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTIONS BY RISK LEVEL
# ─────────────────────────────────────────────────────────────────────────────
def get_solutions(pct):
    if pct >= 85:
        color = "#ff1744"
        solutions = [
            ("🚨 IMMEDIATE DIGITAL DETOX", "#ff1744",
             "Take a 7-day complete break from all non-essential screens. Use this time to rediscover offline hobbies. Studies show even a short detox significantly resets dopamine pathways."),
            ("🧠 SEEK PROFESSIONAL HELP", "#ff4444",
             "Your addiction score is critical. Consult a therapist or counselor specializing in digital addiction. Cognitive Behavioral Therapy (CBT) has 80%+ success rate for screen addiction."),
            ("⏰ ENFORCE HARD SCREEN LIMITS", "#ff6666",
             "Use parental controls or apps like Screen Time / Digital Wellbeing to hard-cap daily usage at 2 hours max. Remove all gaming & social apps from your primary device immediately."),
            ("😴 FIX YOUR SLEEP — NO SCREENS 2 HRS BEFORE BED", "#ff8888",
             "Your pre-bed screen usage is critically high. Place your phone in a different room before sleeping. Replace with reading, journaling, or breathing exercises."),
            ("🏃 MANDATORY DAILY PHYSICAL ACTIVITY", "#ffaaaa",
             "Commit to 60 mins of exercise daily — it is scientifically proven to reduce compulsive screen urges by releasing natural dopamine. Start with a daily walk or join a sport."),
            ("👨‍👩‍👧 FAMILY INTERVENTION & ACCOUNTABILITY PARTNER", "#ffcccc",
             "Share your results with a trusted family member. Set up weekly check-ins. Having accountability reduces relapse by 65%."),
        ]
    elif pct >= 65:
        color = "#ff6d00"
        solutions = [
            ("📵 APP USAGE LIMITS — STRICT SCHEDULE", "#ff6d00",
             "Set hard daily limits: Social Media ≤ 45 min, Gaming ≤ 1 hr, total phone ≤ 3 hrs. Use built-in Screen Time (iPhone) or Digital Wellbeing (Android) right now."),
            ("📚 REPLACE GAMING/SOCIAL TIME WITH SKILLS", "#ff8c00",
             "Convert 1 hour of gaming into learning a new skill — coding, music, drawing. Productive screen time significantly lowers addiction scores over 4–6 weeks."),
            ("😴 NO PHONE IN BEDROOM AFTER 10 PM", "#ffa726",
             "Your before-bed screen habits are harming sleep quality. Invest in a physical alarm clock, charge your phone outside the bedroom. Aim for 8+ hrs sleep."),
            ("🧘 MINDFULNESS & STRESS MANAGEMENT", "#ffb74d",
             "High anxiety/depression correlates with screen overuse. Practice 10–15 min of daily meditation (Headspace, Calm). Reducing stress reduces compulsive checking."),
            ("📅 WEEKLY SCREEN-FREE DAY", "#ffcc80",
             "Pick one day per week (e.g., Sunday) as a complete screen-free day. Fill it with outdoor activities, sports, or social meetups offline."),
        ]
    elif pct >= 45:
        color = "#ffd600"
        solutions = [
            ("⏱️ USE THE 20-20-20 RULE", "#ffd600",
             "Every 20 mins of screen time, look at something 20 feet away for 20 seconds. Set a timer. This breaks the compulsive scroll cycle and protects your eyes."),
            ("📱 TURN OFF ALL NON-ESSENTIAL NOTIFICATIONS", "#ffe033",
             "Disable all social media, game, and news notifications. Research shows notifications are the #1 trigger for compulsive phone checking. Keep only calls/SMS."),
            ("🌿 SCHEDULE OUTDOOR BREAKS", "#ffe566",
             "Take a 15-minute outdoor break every 2 hours of screen use. Nature exposure for just 20 mins/day reduces screen craving significantly."),
            ("📖 SWAP SOCIAL MEDIA FOR READING", "#ffed99",
             "Replace 30 mins of social media with reading a book or long-form article. This builds focus and gives your brain the stimulation it seeks without addiction risk."),
        ]
    elif pct >= 25:
        color = "#00c853"
        solutions = [
            ("✅ MAINTAIN HEALTHY SCREEN HABITS", "#00c853",
             "You're doing well! Keep your phone usage balanced. Continue prioritizing sleep, exercise, and offline socializing to maintain this healthy level."),
            ("📊 TRACK YOUR SCREEN TIME WEEKLY", "#33d175",
             "Use your phone's built-in screen time tracker to monitor weekly trends. Catching any upward creep early prevents future addiction."),
            ("🎯 SET A PERSONAL SCREEN BUDGET", "#66da98",
             "Set a daily screen budget per app category. Having an intentional limit keeps you in control even when stress increases usage temporarily."),
        ]
    else:
        color = "#00bcd4"
        solutions = [
            ("🌟 EXCELLENT SCREEN HEALTH!", "#00bcd4",
             "Your screen usage is minimal and healthy. You're in the top tier of digital wellbeing. Keep up your great habits!"),
            ("💡 SHARE YOUR HABITS", "#29d0e0",
             "Consider sharing your healthy screen habits with friends and family. Peer influence is powerful — you can help others reduce their addiction."),
            ("📈 STAY CONSISTENT", "#66dceb",
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
    if pct >= 85:   return "SEVERE ADDICTION",  "risk-badge-severe",  "#ff1744"
    elif pct >= 65: return "HIGH RISK",          "risk-badge-high",    "#ff6d00"
    elif pct >= 45: return "MODERATE RISK",      "risk-badge-moderate","#ffd600"
    elif pct >= 25: return "LOW RISK",           "risk-badge-low",     "#00c853"
    else:           return "MINIMAL RISK",       "risk-badge-minimal", "#00bcd4"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
(knn, svm, xgb, rf, scaler, feature_cols,
 gender_classes, purpose_classes, grade_order,
 model_scores, feat_imp, df_raw) = train_models()


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📱 MULTISCREEN ADDICTION DETECTOR 💻</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-POWERED TEEN DIGITAL WELLNESS ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

# Model accuracy pills
acc_cols = st.columns(4)
model_labels = [("🤖 KNN", "KNN"), ("⚡ SVM", "SVM"), ("🚀 XGBoost", "XGBoost"), ("🌲 Random Forest", "Random Forest")]
for col, (label, key) in zip(acc_cols, model_labels):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:#00d4ff">{model_scores[key]:.1f}%</div>
            <div class="metric-label">R² Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  PREDICT YOUR ADDICTION", "📊  DATASET INSIGHTS", "🤖  MODEL PERFORMANCE"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📝 Enter Your Details</div>', unsafe_allow_html=True)

    # ── SIDEBAR INPUTS ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='font-family:Orbitron,monospace;font-size:1rem;font-weight:700;
        color:#00d4ff;letter-spacing:2px;text-align:center;padding:0.5rem 0 1rem 0;
        border-bottom:1px solid #7b2fff44;margin-bottom:1rem;'>
        ⚙️ INPUT PANEL
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**👤 Personal Info**")
        age          = st.slider("Age", 13, 18, 16)
        gender       = st.selectbox("Gender", ["Male", "Female", "Other"])
        grade        = st.selectbox("School Grade", ["6th","7th","8th","9th","10th","11th","12th"], index=5)

        st.markdown("---")
        st.markdown("**📱 Phone Usage**")
        daily_usage   = st.slider("Daily Phone Usage (hrs)", 0.0, 12.0, 6.0, 0.5)
        phone_checks  = st.slider("Phone Checks Per Day", 10, 300, 80)
        apps_used     = st.slider("Apps Used Daily", 1, 30, 10)
        stb_phone     = st.slider("Phone Screen Before Bed (hrs)", 0.0, 3.0, 1.0, 0.1)
        weekend_usage = st.slider("Weekend Phone Usage (hrs)", 0.0, 15.0, 6.0, 0.5)
        purpose       = st.selectbox("Primary Phone Purpose", ["Browsing","Gaming","Education","Social Media","Other"])

        st.markdown("**📱 Phone Time Breakdown**")
        t_social  = st.slider("Social Media (hrs)", 0.0, 6.0, 1.5, 0.1)
        t_gaming  = st.slider("Gaming on Phone (hrs)", 0.0, 5.0, 1.0, 0.1)
        t_edu     = st.slider("Education on Phone (hrs)", 0.0, 4.0, 0.8, 0.1)

        st.markdown("---")
        st.markdown("**💻 Laptop Usage**")
        lap_study  = st.slider("Laptop Study (hrs/day)", 0.0, 6.0, 2.0, 0.1)
        lap_gaming = st.slider("Laptop Gaming/Timepass (hrs/day)", 0.0, 6.0, 1.0, 0.1)
        lap_bed    = st.slider("Laptop Before Bed (hrs)", 0.0, 3.0, 0.8, 0.1)

        st.markdown("---")
        st.markdown("**🏃 Health & Lifestyle**")
        sleep    = st.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.5)
        exercise = st.slider("Exercise (hrs/day)", 0.0, 4.0, 0.5, 0.1)
        academic = st.slider("Academic Performance (50–100)", 50, 100, 72)
        social_i = st.slider("Social Interactions (1–10)", 1, 10, 5)
        family_c = st.slider("Family Communication (1–10)", 1, 10, 5)

        st.markdown("**🧠 Mental Health**")
        anxiety    = st.slider("Anxiety Level (1–10)", 1, 10, 5)
        depression = st.slider("Depression Level (1–10)", 1, 10, 4)
        self_est   = st.slider("Self Esteem (1–10)", 1, 10, 6)

        st.markdown("**🔒 Parental Control**")
        parental = st.radio("Parental Control Active?", [0, 1], format_func=lambda x: "✅ Yes" if x else "❌ No")

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
        pulse_class = "pulse" if ensemble_pct >= 85 else ""
        st.markdown(f"""
        <div style='text-align:center; padding:1.5rem 0;'>
            <div style='color:#8888bb;font-family:Orbitron,monospace;font-size:0.8rem;
            letter-spacing:4px;'>ENSEMBLE ADDICTION SCORE</div>
            <div class="big-percent {pulse_class}" style='color:{risk_color};
            text-shadow: 0 0 30px {risk_color}88;'>
                {ensemble_pct:.1f}%
            </div>
            <div class='risk-badge {risk_class}'>{risk_text}</div>
        </div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ensemble_pct,
            number={'suffix': '%', 'font': {'size': 40, 'color': risk_color, 'family': 'Orbitron'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'color': '#8888bb'}, 'tickcolor': '#8888bb'},
                'bar':  {'color': risk_color, 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 25],  'color': 'rgba(0,188,212,0.13)'},
                    {'range': [25, 45], 'color': 'rgba(0,200,83,0.13)'},
                    {'range': [45, 65], 'color': 'rgba(255,214,0,0.13)'},
                    {'range': [65, 85], 'color': 'rgba(255,109,0,0.13)'},
                    {'range': [85, 100],'color': 'rgba(255,23,68,0.13)'},
                ],
                'threshold': {'line': {'color': '#ffffff', 'width': 3},
                              'thickness': 0.8, 'value': ensemble_pct}
            },
            title={'text': "ADDICTION LEVEL", 'font': {'size': 14, 'color': '#8888bb', 'family': 'Orbitron'}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0ff', height=280, margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        model_names = list(pct_preds.keys())
        model_vals  = list(pct_preds.values())
        bar_clrs    = [risk_color if v >= 85 else '#ff6d00' if v >= 65
                       else '#ffd600' if v >= 45 else '#00c853' for v in model_vals]

        fig_bar = go.Figure(go.Bar(
            x=model_names, y=model_vals,
            marker=dict(color=bar_clrs, line=dict(color='rgba(255,255,255,0.3)', width=1)),
            text=[f'{v:.1f}%' for v in model_vals], textposition='outside',
            textfont=dict(family='Orbitron', size=13, color='white')
        ))
        fig_bar.add_hline(y=ensemble_pct, line_dash="dash", line_color="#ffffff",
                          annotation_text=f"Ensemble: {ensemble_pct:.1f}%",
                          annotation_font_color="#ffffff")
        fig_bar.update_layout(
            title=dict(text="PREDICTION BY EACH MODEL", font=dict(family='Orbitron', size=12, color='#8888bb')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,62,0.5)',
            font_color='#e0e0ff', height=280,
            yaxis=dict(range=[0, 110], gridcolor='#333355', title='Addiction %'),
            xaxis=dict(gridcolor='#333355'),
            margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

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
            fillcolor=risk_color,
            line=dict(color=risk_color, width=2),
            marker=dict(color=risk_color, size=6)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(26,26,62,0.5)',
                radialaxis=dict(visible=True, range=[0, 10],
                                gridcolor='#333355', tickfont=dict(color='#666688')),
                angularaxis=dict(gridcolor='#333355', tickfont=dict(color='#c8c8e8', size=11))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(text="SCREEN TIME PROFILE", font=dict(family='Orbitron', size=11, color='#8888bb')),
            height=280, margin=dict(t=50, b=10, l=40, r=40),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        for model, pct in pct_preds.items():
            clr = risk_info(pct)[2]
            st.markdown(f"""
            <div style='background:rgba(26,26,62,0.7);border:1px solid {clr}44;
            border-radius:10px;padding:0.6rem 1rem;margin:0.4rem 0;
            display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-family:Orbitron,monospace;font-size:0.75rem;
                color:#8888bb;letter-spacing:1px;'>{model}</span>
                <span style='font-family:Orbitron,monospace;font-size:1.2rem;
                font-weight:700;color:{clr};'>{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

    # ── SOLUTIONS ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="section-header" style='color:{risk_color_sol};'>
        💡 PERSONALIZED SOLUTIONS — {risk_text}
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
    st.markdown('<div class="section-header">📊 YOUR SCREEN TIME BREAKDOWN</div>', unsafe_allow_html=True)

    pie_labels = ['Social Media (Phone)', 'Gaming (Phone)', 'Education (Phone)',
                  'Laptop Study', 'Laptop Gaming', 'Laptop Before Bed', 'Other Phone Time']
    other_phone = max(0, daily_usage - t_social - t_gaming - t_edu)
    pie_values  = [t_social, t_gaming, t_edu, lap_study, lap_gaming, lap_bed, other_phone]
    pie_colors  = ['#ff6ec7', '#ff4444', '#00d4ff', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    pie_values_clean = [max(0, v) for v in pie_values]

    if sum(pie_values_clean) > 0:
        fig_pie = go.Figure(go.Pie(
            labels=pie_labels, values=pie_values_clean, hole=0.5,
            marker=dict(colors=pie_colors, line=dict(color='#0a0a1a', width=2)),
            textfont=dict(family='Rajdhani', size=12),
            hovertemplate='<b>%{label}</b><br>%{value:.1f} hrs/day<br>%{percent}<extra></extra>'
        ))
        fig_pie.add_annotation(
            text=f"{sum(pie_values_clean):.1f} hrs<br>total/day",
            x=0.5, y=0.5, font=dict(family='Orbitron', size=13, color='white'),
            showarrow=False
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0ff',
            height=350, margin=dict(t=20, b=20, l=0, r=0),
            legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📈 Dataset Overview</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    for col, (val, label, color) in zip(
        [kpi1, kpi2, kpi3, kpi4],
        [(len(df_raw), "TOTAL RECORDS", "#00d4ff"),
         (len(df_raw.columns), "FEATURES", "#7b2fff"),
         (f"{df_raw['Addiction_Level'].mean():.1f}/10", "AVG ADDICTION", "#ff6ec7"),
         (f"{df_raw['Daily_Usage_Hours'].mean():.1f}h",  "AVG DAILY USE", "#ff6d00")]
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
                                 color_discrete_sequence=['#7b2fff'],
                                 title='Addiction Level Distribution')
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(26,26,62,0.5)',
                                font_color='#e0e0ff', height=300,
                                title_font=dict(family='Orbitron', size=12, color='#8888bb'))
        fig_hist.update_xaxes(gridcolor='#333355')
        fig_hist.update_yaxes(gridcolor='#333355')
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(df_raw,
                          x='Phone_Usage_Purpose', y='Addiction_Level',
                          color='Phone_Usage_Purpose',
                          color_discrete_sequence=['#ff6ec7','#7b2fff','#00d4ff','#2ecc71','#ff6d00'],
                          title='Addiction by Phone Purpose')
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(26,26,62,0.5)',
                               font_color='#e0e0ff', height=300, showlegend=False,
                               title_font=dict(family='Orbitron', size=12, color='#8888bb'))
        fig_box.update_xaxes(gridcolor='#333355')
        fig_box.update_yaxes(gridcolor='#333355')
        st.plotly_chart(fig_box, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_sc1 = px.scatter(df_raw, x='Daily_Usage_Hours', y='Addiction_Level',
                              color='Addiction_Level', color_continuous_scale='plasma',
                              opacity=0.5, title='Daily Usage vs Addiction')
        fig_sc1.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(26,26,62,0.5)',
                               font_color='#e0e0ff', height=300,
                               title_font=dict(family='Orbitron', size=12, color='#8888bb'))
        fig_sc1.update_xaxes(gridcolor='#333355')
        fig_sc1.update_yaxes(gridcolor='#333355')
        st.plotly_chart(fig_sc1, use_container_width=True)

    with c4:
        fig_sc2 = px.scatter(df_raw, x='Sleep_Hours', y='Addiction_Level',
                              color='Anxiety_Level', color_continuous_scale='reds',
                              opacity=0.5, title='Sleep Hours vs Addiction (colored by Anxiety)')
        fig_sc2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(26,26,62,0.5)',
                               font_color='#e0e0ff', height=300,
                               title_font=dict(family='Orbitron', size=12, color='#8888bb'))
        fig_sc2.update_xaxes(gridcolor='#333355')
        fig_sc2.update_yaxes(gridcolor='#333355')
        st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown('<div class="section-header">💻 Laptop Usage Analysis</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    laptop_cols_plot = [
        ('Laptop_Study_Hours', '#2ecc71', 'Laptop Study vs Addiction'),
        ('Laptop_Gaming_TimePass_Hours', '#e74c3c', 'Laptop Gaming vs Addiction'),
        ('Laptop_Usage_Before_Bed_Hours', '#9b59b6', 'Laptop Before Bed vs Addiction'),
    ]
    for col, (xcol, clr, title) in zip([lc1, lc2, lc3], laptop_cols_plot):
        with col:
            fig_l = px.scatter(df_raw, x=xcol, y='Addiction_Level', opacity=0.4,
                                color_discrete_sequence=[clr], title=title)
            fig_l.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                 plot_bgcolor='rgba(26,26,62,0.5)',
                                 font_color='#e0e0ff', height=280,
                                 title_font=dict(family='Orbitron', size=10, color='#8888bb'),
                                 margin=dict(t=40,b=20,l=20,r=20))
            fig_l.update_xaxes(gridcolor='#333355')
            fig_l.update_yaxes(gridcolor='#333355')
            st.plotly_chart(fig_l, use_container_width=True)

    st.markdown('<div class="section-header">👥 Demographics</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        gender_avg = df_raw.groupby('Gender')['Addiction_Level'].mean().reset_index()
        fig_g = px.bar(gender_avg, x='Gender', y='Addiction_Level',
                        color='Gender', color_discrete_sequence=['#ff6ec7','#7b2fff','#00d4ff'],
                        title='Avg Addiction by Gender')
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,62,0.5)',
                             font_color='#e0e0ff', height=280, showlegend=False,
                             title_font=dict(family='Orbitron', size=12, color='#8888bb'))
        fig_g.update_xaxes(gridcolor='#333355'); fig_g.update_yaxes(gridcolor='#333355')
        st.plotly_chart(fig_g, use_container_width=True)

    with d2:
        grade_avg = df_raw.groupby('School_Grade')['Addiction_Level'].mean().reset_index()
        grade_order_list = ['6th','7th','8th','9th','10th','11th','12th']
        grade_avg['School_Grade'] = pd.Categorical(grade_avg['School_Grade'],
                                                    categories=grade_order_list, ordered=True)
        grade_avg = grade_avg.sort_values('School_Grade')
        fig_gr = px.line(grade_avg, x='School_Grade', y='Addiction_Level',
                          markers=True, title='Avg Addiction by School Grade',
                          color_discrete_sequence=['#00d4ff'])
        fig_gr.update_traces(line=dict(width=3), marker=dict(size=10, color='#ff6ec7'))
        fig_gr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,62,0.5)',
                              font_color='#e0e0ff', height=280,
                              title_font=dict(family='Orbitron', size=12, color='#8888bb'))
        fig_gr.update_xaxes(gridcolor='#333355'); fig_gr.update_yaxes(gridcolor='#333355')
        st.plotly_chart(fig_gr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 Model Accuracy Comparison</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, (model, score) in zip([mc1, mc2, mc3, mc4], model_scores.items()):
        clr = "#2ecc71" if score >= 70 else "#ffd600" if score >= 50 else "#ff6d00"
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
            color=['#3498db','#e67e22','#e74c3c','#2ecc71'],
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f'{v:.1f}%' for v in model_scores.values()],
        textposition='outside', textfont=dict(family='Orbitron', size=14, color='white')
    ))
    fig_acc.update_layout(
        title=dict(text='MODEL R² ACCURACY (%)', font=dict(family='Orbitron', size=13, color='#8888bb')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,62,0.5)',
        font_color='#e0e0ff', height=350,
        yaxis=dict(range=[0, 110], gridcolor='#333355', title='Accuracy %'),
        xaxis=dict(gridcolor='#333355'),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown('<div class="section-header">🌟 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    top_n = 20
    fi_df = feat_imp.head(top_n).reset_index()
    fi_df.columns = ['Feature', 'Importance']
    fi_colors = ['#e74c3c' if i < 5 else '#ff6d00' if i < 10 else '#3498db'
                 for i in range(len(fi_df))]

    fig_fi = go.Figure(go.Bar(
        x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
        marker=dict(color=fi_colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
        text=[f'{v:.4f}' for v in fi_df['Importance']],
        textposition='outside', textfont=dict(size=10, color='white')
    ))
    fig_fi.update_layout(
        title=dict(text=f'TOP {top_n} FEATURES  (🔴=Top5  🟠=Top10  🔵=Rest)',
                   font=dict(family='Orbitron', size=12, color='#8888bb')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,62,0.5)',
        font_color='#e0e0ff', height=600,
        yaxis=dict(autorange='reversed', gridcolor='#333355', tickfont=dict(size=11)),
        xaxis=dict(gridcolor='#333355', title='Importance Score'),
        margin=dict(t=50, b=20, l=200, r=80)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    📱 MULTISCREEN ADDICTION DETECTOR  •  POWERED BY KNN | SVM | XGBOOST | RANDOM FOREST
    <br>TRAINED ON 3000 TEEN RECORDS  •  28 FEATURES  •  AI-POWERED DIGITAL WELLNESS ANALYSIS
</div>
""", unsafe_allow_html=True)
