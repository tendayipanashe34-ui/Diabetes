"""
Diabetes Prediction — Streamlit Application
Frontend  : Streamlit
Auth      : Session-based (username + hashed password)
Database  : SQLite (users, predictions tables)
Model     : Deep ANN logistic scoring (mirrors notebook)
"""

import streamlit as st
import sqlite3
import hashlib
import os
import json
import math
from datetime import datetime
from pathlib import Path

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetes Web Predictior",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Database Setup ─────────────────────────────────────────────────────────

DB_PATH = "diabetes.db"

def init_db():
    """Initialize database with users and predictions tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            full_name     TEXT,
            email         TEXT,
            role          TEXT    DEFAULT 'patient',
            created_at    TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            gender        TEXT,
            chol          REAL,   chol_label      TEXT,
            stab_glu      REAL,   stab_glu_label  TEXT,
            hdl           REAL,   hdl_label       TEXT,
            ratio         REAL,   ratio_label     TEXT,
            age           REAL,   age_label       TEXT,
            height        REAL,   height_label    TEXT,
            weight        REAL,   weight_label    TEXT,
            waist         REAL,   waist_label     TEXT,
            hip           REAL,   hip_label       TEXT,
            bp1s          REAL,   bp1s_label      TEXT,
            bp1d          REAL,   bp1d_label      TEXT,
            time_ppn      REAL,   time_ppn_label  TEXT,
            risk_score    REAL,
            probability   REAL,
            prediction    TEXT,
            flagged_count INTEGER,
            flagged_reasons TEXT,
            created_at    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
    
    # Seed demo accounts
    _seed_users()

def _seed_users():
    """Seed database with demo accounts."""
    seeds = [
        ("admin",   "admin123",   "Administrator",  "admin@clinic.com",  "admin"),
        ("doctor1", "doctor123",  "Dr. Jane Smith", "jsmith@clinic.com", "doctor"),
        ("patient1","patient123", "John Doe",       "jdoe@clinic.com",   "patient"),
    ]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for uname, pw, name, email, role in seeds:
        try:
            cursor.execute(
                "INSERT INTO users (username,password_hash,full_name,email,role) VALUES (?,?,?,?,?)",
                (uname, _hash(pw), name, email, role)
            )
        except sqlite3.IntegrityError:
            pass
    
    conn.commit()
    conn.close()

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ─── Auth Functions ────────────────────────────────────────────────────────

def _hash(pw):
    """Hash password using SHA256."""
    return hashlib.sha256(pw.encode()).hexdigest()

def login(username, password):
    """Authenticate user and return user data."""
    conn = get_db()
    cursor = conn.cursor()
    user = cursor.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, _hash(password))
    ).fetchone()
    conn.close()
    
    if user:
        return dict(user)
    return None

def register(username, password, full_name, email):
    """Register new user."""
    if not username or not password:
        return False, "Username and password are required"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO users (username,password_hash,full_name,email) VALUES (?,?,?,?)",
            (username, _hash(password), full_name, email)
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists"

# ─── Prediction Model ──────────────────────────────────────────────────────

def _zscore(v, mean, std):
    return (v - mean) / std

def _sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

SCALER = {
    "stab_glu": (107, 45),  "ratio":  (4.5, 1.5), "hdl":   (50,  17),
    "age":      (46,  16),  "chol":   (207, 45),  "bp1s":  (136, 23),
    "bp1d":     (83,  14),  "weight": (177, 44),  "waist": (36,  6),
    "time_ppn": (90,  60),  "height": (66,  4),   "hip":   (40,  5),
}

def compute_probability(v, gender):
    """Compute diabetes risk probability."""
    z = {k: _zscore(v[k], *SCALER[k]) for k in SCALER}
    bmi = (v["weight"] * 703) / (v["height"] ** 2)
    z_bmi = _zscore(bmi, 29, 6)
    whr = v["waist"] / v["hip"]
    z_whr = _zscore(whr, 0.87, 0.1)
    gender_bias = 0.15 if gender == "male" else 0.0

    logit = (
        -5.2
        + 3.2 * z["stab_glu"]
        + 1.4 * z["ratio"]
        - 1.2 * z["hdl"]
        + 1.0 * z["age"]
        + 0.9 * z_bmi
        + 0.7 * z_whr
        + 0.5 * z["chol"]
        + 0.4 * z["bp1s"]
        + 0.3 * z["weight"]
        + 0.2 * z["bp1d"]
        + 0.2 * z["time_ppn"]
        + 0.1 * z["waist"]
        + gender_bias
    )
    return _sigmoid(logit)

def evaluate_risk_flags(data):
    """Evaluate all risk flags and return count and detailed reasons."""
    flags = []
    
    # Glucose risk
    if data["stab_glu"] >= 126:
        flags.append({"condition": "High Fasting Glucose", 
                     "value": data["stab_glu"], 
                     "threshold": "≥126 mg/dL",
                     "severity": "High",
                     "explanation": "Elevated blood glucose levels indicate prediabetes or diabetes"})
    elif data["stab_glu"] >= 100:
        flags.append({"condition": "Borderline Glucose", 
                     "value": data["stab_glu"], 
                     "threshold": "100-125 mg/dL",
                     "severity": "Moderate",
                     "explanation": "Impaired fasting glucose - prediabetes stage"})
    
    # Cholesterol ratio risk
    if data["ratio"] >= 7:
        flags.append({"condition": "Poor Cholesterol Ratio", 
                     "value": data["ratio"], 
                     "threshold": "≥7",
                     "severity": "High",
                     "explanation": "High LDL to HDL ratio increases cardiovascular and diabetes risk"})
    elif data["ratio"] >= 5:
        flags.append({"condition": "Borderline Cholesterol Ratio", 
                     "value": data["ratio"], 
                     "threshold": "5-7",
                     "severity": "Moderate",
                     "explanation": "Moderately elevated cholesterol ratio"})
    
    # HDL risk
    if data["hdl"] < 40:
        flags.append({"condition": "Low HDL Cholesterol", 
                     "value": data["hdl"], 
                     "threshold": "<40 mg/dL",
                     "severity": "High",
                     "explanation": "Low 'good' cholesterol increases metabolic syndrome risk"})
    elif data["hdl"] < 50:
        flags.append({"condition": "Borderline HDL", 
                     "value": data["hdl"], 
                     "threshold": "40-50 mg/dL",
                     "severity": "Moderate",
                     "explanation": "Moderately low HDL levels"})
    
    # BMI risk
    bmi = (data["weight"] * 703) / (data["height"] ** 2)
    if bmi >= 30:
        flags.append({"condition": "Obesity", 
                     "value": f"{bmi:.1f}", 
                     "threshold": "BMI ≥30",
                     "severity": "High",
                     "explanation": "Obesity is a major risk factor for insulin resistance and type 2 diabetes"})
    elif bmi >= 25:
        flags.append({"condition": "Overweight", 
                     "value": f"{bmi:.1f}", 
                     "threshold": "BMI 25-30",
                     "severity": "Moderate",
                     "explanation": "Excess weight increases diabetes risk"})
    
    # Blood pressure risk
    if data["bp1s"] >= 140:
        flags.append({"condition": "High Systolic BP", 
                     "value": data["bp1s"], 
                     "threshold": "≥140 mmHg",
                     "severity": "High",
                     "explanation": "Hypertension is commonly associated with diabetes"})
    elif data["bp1s"] >= 130:
        flags.append({"condition": "Elevated Systolic BP", 
                     "value": data["bp1s"], 
                     "threshold": "130-139 mmHg",
                     "severity": "Moderate",
                     "explanation": "Pre-hypertension stage"})
    
    # Waist circumference risk
    if data["waist"] >= 40:
        flags.append({"condition": "High Waist Circumference", 
                     "value": data["waist"], 
                     "threshold": "≥40 inches (men) or ≥35 inches (women)",
                     "severity": "High",
                     "explanation": "Central obesity indicates higher visceral fat and insulin resistance"})
    elif data["waist"] >= 35:
        flags.append({"condition": "Elevated Waist Circumference", 
                     "value": data["waist"], 
                     "threshold": "35-40 inches",
                     "severity": "Moderate",
                     "explanation": "Increased abdominal fat distribution"})
    
    # Age risk
    if data["age"] >= 60:
        flags.append({"condition": "Advanced Age", 
                     "value": data["age"], 
                     "threshold": "≥60 years",
                     "severity": "High",
                     "explanation": "Diabetes risk increases significantly with age"})
    elif data["age"] >= 45:
        flags.append({"condition": "Middle Age", 
                     "value": data["age"], 
                     "threshold": "45-60 years",
                     "severity": "Moderate",
                     "explanation": "Age-related metabolic changes increase risk"})
    
    # Waist-to-hip ratio risk
    whr = data["waist"] / data["hip"]
    if whr > 0.9:
        flags.append({"condition": "High Waist-to-Hip Ratio", 
                     "value": f"{whr:.2f}", 
                     "threshold": ">0.9 (men) or >0.85 (women)",
                     "severity": "High",
                     "explanation": "Upper body fat distribution pattern increases metabolic risk"})
    
    # Total cholesterol risk
    if data["chol"] >= 240:
        flags.append({"condition": "High Total Cholesterol", 
                     "value": data["chol"], 
                     "threshold": "≥240 mg/dL",
                     "severity": "High",
                     "explanation": "Hypercholesterolemia increases cardiovascular complications"})
    
    return flags

def save_prediction(user_id, data, prob, prediction, flags):
    """Save prediction to database."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Convert flags to JSON for storage
    flags_json = json.dumps([f["condition"] for f in flags])
    
    cursor.execute(
        """INSERT INTO predictions (
            user_id, gender,
            chol, chol_label, stab_glu, stab_glu_label,
            hdl, hdl_label, ratio, ratio_label,
            age, age_label, height, height_label,
            weight, weight_label, waist, waist_label,
            hip, hip_label, bp1s, bp1s_label,
            bp1d, bp1d_label, time_ppn, time_ppn_label,
            risk_score, probability, prediction, flagged_count, flagged_reasons
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            user_id, data["gender"],
            data["chol"],     data.get("chol_label",""),
            data["stab_glu"], data.get("stab_glu_label",""),
            data["hdl"],      data.get("hdl_label",""),
            data["ratio"],    data.get("ratio_label",""),
            data["age"],      data.get("age_label",""),
            data["height"],   data.get("height_label",""),
            data["weight"],   data.get("weight_label",""),
            data["waist"],    data.get("waist_label",""),
            data["hip"],      data.get("hip_label",""),
            data["bp1s"],     data.get("bp1s_label",""),
            data["bp1d"],     data.get("bp1d_label",""),
            data["time_ppn"], data.get("time_ppn_label",""),
            round(prob * 100, 1), round(prob, 4), prediction, len(flags), flags_json,
        )
    )
    conn.commit()
    conn.close()

def get_user_history(user_id, role):
    """Fetch user prediction history."""
    conn = get_db()
    cursor = conn.cursor()
    
    if role in ("admin", "doctor"):
        rows = cursor.execute("""
            SELECT p.*, u.username, u.full_name
            FROM predictions p JOIN users u ON p.user_id=u.id
            ORDER BY p.created_at DESC LIMIT 100
        """).fetchall()
    else:
        rows = cursor.execute("""
            SELECT p.*, u.username, u.full_name
            FROM predictions p JOIN users u ON p.user_id=u.id
            WHERE p.user_id=?
            ORDER BY p.created_at DESC LIMIT 50
        """, (user_id,)).fetchall()
    
    conn.close()
    return [dict(r) for r in rows]

def get_stats(user_id, role):
    """Get prediction statistics."""
    conn = get_db()
    cursor = conn.cursor()
    
    scope = "" if role in ("admin","doctor") else f"WHERE user_id={user_id}"
    row = cursor.execute(f"""
        SELECT
          COUNT(*) total,
          SUM(CASE WHEN prediction='Diabetic' THEN 1 ELSE 0 END) diabetic,
          SUM(CASE WHEN prediction='Non-Diabetic' THEN 1 ELSE 0 END) non_diabetic,
          ROUND(AVG(risk_score),1) avg_risk
        FROM predictions {scope}
    """).fetchone()
    
    conn.close()
    return dict(row) if row else {}

# ─── Streamlit Session State ──────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.full_name = None
    st.session_state.role = None

# ─── Initialize Database ──────────────────────────────────────────────────

if not Path(DB_PATH).exists():
    init_db()

# ─── Main App ──────────────────────────────────────────────────────────────

if not st.session_state.authenticated:
    # Authentication Page
    st.title("🩺 Diabetes Prediction System")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            user = login(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user["id"]
                st.session_state.username = user["username"]
                st.session_state.full_name = user["full_name"]
                st.session_state.role = user["role"]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="reg_username")
        new_password = st.text_input("Password", type="password", key="reg_password")
        full_name = st.text_input("Full Name", key="reg_fullname")
        email = st.text_input("Email", key="reg_email")
        
        if st.button("Create Account"):
            success, message = register(new_username, new_password, full_name, email)
            if success:
                st.success(message)
            else:
                st.error(message)

else:
    # Main Application
    st.title("🩺 Diabetes Prediction System")
    
    # Sidebar
    with st.sidebar:
        st.write(f"**User:** {st.session_state.full_name}")
        st.write(f"**Role:** {st.session_state.role.capitalize()}")
        
        page = st.radio("Navigation", ["Dashboard", "Predict", "History", "Statistics", "Logout"])
    
    if page == "Logout":
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.full_name = None
        st.session_state.role = None
        st.success("Logged out successfully!")
        st.rerun()
    
    elif page == "Dashboard":
        st.subheader("Dashboard")
        st.write(f"Welcome back, {st.session_state.full_name}!")
        
        # Display quick stats
        stats = get_stats(st.session_state.user_id, st.session_state.role)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", stats.get("total", 0))
        with col2:
            st.metric("Diabetic Cases", stats.get("diabetic", 1))
        with col3:
            st.metric("Non-Diabetic Cases", stats.get("non_diabetic", 0))
        with col4:
            avg_risk = stats.get('avg_risk')
            if avg_risk is None:
                avg_risk = 0
            st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
            
    
    elif page == "Predict":
        st.subheader("Diabetes Risk Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=45)
            
            st.write("**Physical Measurements**")
            height = st.number_input("Height (inches)", min_value=36.0, max_value=84.0, value=66.0)
            weight = st.number_input("Weight (lbs)", min_value=70.0, max_value=400.0, value=177.0)
            waist = st.number_input("Waist (inches)", min_value=20.0, max_value=60.0, value=36.0)
            hip = st.number_input("Hip (inches)", min_value=20.0, max_value=60.0, value=40.0)
        
        with col2:
            st.write("**Metabolic Markers**")
            chol = st.number_input("Total Cholesterol (mg/dL)", min_value=50.0, max_value=400.0, value=207.0)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=10.0, max_value=200.0, value=50.0)
            ratio = st.number_input("Chol/HDL Ratio", min_value=1.0, max_value=15.0, value=4.5)
            stab_glu = st.number_input("Fasting Glucose (mg/dL)", min_value=50.0, max_value=300.0, value=107.0)
            
            st.write("**Blood Pressure & Timing**")
            bp1s = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=250.0, value=136.0)
            bp1d = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=150.0, value=83.0)
            time_ppn = st.number_input("Time Post Meal (min)", min_value=30.0, max_value=720.0, value=90.0)
        
        if st.button("🔮 Predict Risk", key="predict_btn"):
            # Prepare data
            data = {
                "gender": gender.lower(),
                "chol": chol,
                "stab_glu": stab_glu,
                "hdl": hdl,
                "ratio": ratio,
                "age": age,
                "height": height,
                "weight": weight,
                "waist": waist,
                "hip": hip,
                "bp1s": bp1s,
                "bp1d": bp1d,
                "time_ppn": time_ppn,
            }
            
            # Compute probability
            prob = compute_probability(data, data["gender"])
            risk_score = round(prob * 100, 1)
            prediction = "Diabetic" if prob >= 0.5 else "Non-Diabetic"
            
            # Evaluate risk flags
            risk_flags = evaluate_risk_flags(data)
            
            # Save to database
            save_prediction(st.session_state.user_id, data, prob, prediction, risk_flags)
            
            # Display results
            st.success("Prediction completed!")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == "Diabetic":
                    st.error(f"**Prediction: {prediction}**")
                else:
                    st.success(f"**Prediction: {prediction}**")
                st.metric("Risk Score", f"{risk_score}%")
                st.metric("Risk Probability", f"{prob:.4f}")
            
            with result_col2:
                bmi = (weight * 703) / (height ** 2)
                st.metric("BMI", f"{bmi:.1f}")
                st.metric("High-Risk Flags", len(risk_flags))
                st.write(f"**WHR:** {waist/hip:.2f}")
            
            # Risk assessment
            if risk_score >= 80:
                st.warning("⚠️ **High Risk** - Immediate medical consultation recommended")
            elif risk_score >= 50:
                st.info("⚠️ **Moderate Risk** - Schedule a checkup with your doctor")
            else:
                st.success("✅ **Low Risk** - Continue healthy lifestyle habits")
            
            # Display detailed flags and causes
            if risk_flags:
                st.subheader("🚨 Detailed Risk Factors & Main Causes")
                st.write("The following factors are contributing to your diabetes risk:")
                
                # Group by severity
                high_risk = [f for f in risk_flags if f["severity"] == "High"]
                moderate_risk = [f for f in risk_flags if f["severity"] == "Moderate"]
                
                if high_risk:
                    st.write("**🔴 High Severity Risk Factors:**")
                    for flag in high_risk:
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 3])
                            with col1:
                                st.write(f"**{flag['condition']}**")
                            with col2:
                                st.write(f"Value: {flag['value']}")
                            with col3:
                                st.write(f"Threshold: {flag['threshold']}")
                            st.caption(f"📌 {flag['explanation']}")
                            st.divider()
                
                if moderate_risk:
                    st.write("**🟡 Moderate Severity Risk Factors:**")
                    for flag in moderate_risk:
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 3])
                            with col1:
                                st.write(f"**{flag['condition']}**")
                            with col2:
                                st.write(f"Value: {flag['value']}")
                            with col3:
                                st.write(f"Threshold: {flag['threshold']}")
                            st.caption(f"📌 {flag['explanation']}")
                            st.divider()
                
                # Lifestyle recommendations based on flags
                st.subheader("💡 Personalized Recommendations")
                recommendations = []
                
                if any(f["condition"] in ["High Fasting Glucose", "Borderline Glucose"] for f in risk_flags):
                    recommendations.append("• Monitor blood glucose regularly and consider reducing carbohydrate intake")
                if any(f["condition"] in ["Obesity", "Overweight"] for f in risk_flags):
                    recommendations.append("• Aim for gradual weight loss through diet and exercise (5-10% of body weight)")
                if any(f["condition"] in ["High Waist Circumference", "Elevated Waist Circumference"] for f in risk_flags):
                    recommendations.append("• Focus on reducing abdominal fat with aerobic exercise and strength training")
                if any(f["condition"] in ["High Systolic BP", "Elevated Systolic BP"] for f in risk_flags):
                    recommendations.append("• Reduce sodium intake, increase physical activity, and consider DASH diet")
                if any("HDL" in f["condition"] for f in risk_flags):
                    recommendations.append("• Increase omega-3 fatty acids (fish, nuts, seeds) and regular exercise to boost HDL")
                if any("Cholesterol Ratio" in f["condition"] for f in risk_flags):
                    recommendations.append("• Improve lipid profile with fiber-rich foods and healthy fats")
                if age >= 45:
                    recommendations.append("• Regular diabetes screening recommended due to age-related risk")
                
                if recommendations:
                    for rec in recommendations:
                        st.write(rec)
                else:
                    st.write("• Continue maintaining healthy lifestyle habits")
                    st.write("• Regular check-ups and monitoring recommended")
            else:
                st.success("🎉 No major risk factors detected! Continue maintaining your healthy lifestyle.")
    
    elif page == "History":
        st.subheader("Prediction History")
        
        history = get_user_history(st.session_state.user_id, st.session_state.role)
        
        if history:
            for pred in history:
                with st.expander(f"📊 {pred['created_at']} - {pred['prediction']} ({pred['risk_score']}%)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**User:** {pred['full_name']}")
                        st.write(f"**Gender:** {pred['gender']}")
                        st.write(f"**Age:** {pred['age']}")
                    with col2:
                        st.write(f"**Risk Score:** {pred['risk_score']}%")
                        st.write(f"**Probability:** {pred['probability']:.4f}")
                        st.write(f"**Flags:** {pred['flagged_count']}")
                    with col3:
                        st.write(f"**Cholesterol:** {pred['chol']}")
                        st.write(f"**Glucose:** {pred['stab_glu']}")
                        st.write(f"**HDL:** {pred['hdl']}")
                    
                    # Show flagged reasons if available
                    if pred.get('flagged_reasons'):
                        st.write("**Risk Factors Identified:**")
                        reasons = json.loads(pred['flagged_reasons'])
                        for reason in reasons:
                            st.write(f"• {reason}")
        else:
            st.info("No prediction history available.")
    
    elif page == "Statistics":
        st.subheader("Statistics & Analytics")
        
        stats = get_stats(st.session_state.user_id, st.session_state.role)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", stats.get("total", 0))
        with col2:
            st.metric("Diabetic Cases", stats.get("diabetic", 1))
        with col3:
            st.metric("Non-Diabetic Cases", stats.get("non_diabetic", 0))
        with col4:
            st.metric("Avg Risk Score", f"{stats.get('avg_risk', 0):.1f}%")
        
        # Calculate percentages
        total = stats.get("total", 1)
        diabetic_pct = (stats.get("diabetic", 0) / total * 100) if total > 0 else 0
        
        st.write(f"**Diabetic Percentage:** {diabetic_pct:.1f}%")
        st.write(f"**Non-Diabetic Percentage:** {100 - diabetic_pct:.1f}%")
