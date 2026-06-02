"""
Diabetes Prediction — Streamlit Application
Frontend  : Streamlit
Auth      : Session-based (username + hashed password)
Database  : SQLite (users, predictions, notifications tables)
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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetes Prediction System",
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
            role          TEXT    NOT NULL,
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
            created_at    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            prediction_id INTEGER,
            title         TEXT,
            message       TEXT,
            recommendations TEXT,
            is_read       INTEGER DEFAULT 0,
            created_at    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
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

def register(username, password, full_name, email, role="patient"):
    """Register new user with explicit role specification."""
    if not username or not password:
        return False, "Username and password are required"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if role not in ("patient", "doctor", "admin"):
        return False, "Invalid role specified"
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO users (username,password_hash,full_name,email,role) VALUES (?,?,?,?,?)",
            (username, _hash(password), full_name, email, role)
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

def save_prediction(user_id, data, prob, prediction, flags):
    """Save prediction to database."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        """INSERT INTO predictions (
            user_id, gender,
            chol, chol_label, stab_glu, stab_glu_label,
            hdl, hdl_label, ratio, ratio_label,
            age, age_label, height, height_label,
            weight, weight_label, waist, waist_label,
            hip, hip_label, bp1s, bp1s_label,
            bp1d, bp1d_label, time_ppn, time_ppn_label,
            risk_score, probability, prediction, flagged_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
            round(prob * 100, 1), round(prob, 4), prediction, flags,
        )
    )
    
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return pred_id

# ─── Notification Functions ────────────────────────────────────────────────
# Add this function to final.py (near your other database functions)

def init_database():
    """Initialize all required database tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create notifications table (the missing one)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            message TEXT NOT NULL,
            type TEXT DEFAULT 'info',
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')
    
    # Create users table (if not exists)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table (if you have one)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            prediction_result INTEGER,
            probability REAL,
            input_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Call this function at the START of your app (after st.set_page_config)
# Add this line in the main section:
# init_database()
def create_notification(user_id, prediction_id, risk_score, prediction, data):
    """Create a structured notification with greeting and recommendations."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get user info
    user = cursor.execute("SELECT full_name FROM users WHERE id=?", (user_id,)).fetchone()
    user_name = user["full_name"] if user else "User"
    
    # Create title and greeting
    if prediction == "Diabetic":
        title = "⚠️ High Diabetes Risk Detected"
        greeting = f"Hello {user_name},\n\nYour recent health assessment shows a high diabetes risk."
    else:
        title = "✅ Low Diabetes Risk"
        greeting = f"Hello {user_name},\n\nGreat news! Your health assessment shows a low diabetes risk."
    
    # Generate recommendations
    recommendations = _generate_recommendations(risk_score, data)
    
    message = f"""{greeting}

**Your Results:**
• Risk Score: {risk_score}%
• Prediction: {prediction}
• Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Please review the recommendations below and consult with your healthcare provider if needed."""
    
    cursor.execute(
        """INSERT INTO notifications 
           (user_id, prediction_id, title, message, recommendations, is_read)
           VALUES (?, ?, ?, ?, ?, 0)""",
        (user_id, prediction_id, title, message, recommendations)
    )
    
    conn.commit()
    conn.close()

def _generate_recommendations(risk_score, data):
    """Generate personalized recommendations based on risk score and health metrics."""
    recommendations = []
    
    if risk_score >= 80:
        recommendations.extend([
            "🚨 IMMEDIATE ACTIONS REQUIRED:",
            "• Schedule an urgent appointment with your doctor",
            "• Get comprehensive blood work and glucose tolerance tests",
            "• Consider endocrinologist referral",
            "• Start daily glucose monitoring",
            "• Review medication options with healthcare provider"
        ])
    elif risk_score >= 50:
        recommendations.extend([
            "⚠️ RECOMMENDED ACTIONS:",
            "• Schedule doctor checkup within 2 weeks",
            "• Get HbA1c test (3-month glucose average)",
            "• Begin home glucose monitoring 2-3 times weekly",
            "• Work with a nutritionist on meal planning",
            "• Start regular exercise program (150 min/week)"
        ])
    else:
        recommendations.extend([
            "✅ MAINTENANCE ACTIONS:",
            "• Continue annual health checkups",
            "• Maintain regular physical activity",
            "• Follow healthy eating habits",
            "• Monitor weight quarterly",
            "• Follow preventive screening recommendations"
        ])
    
    # Add dietary recommendations
    recommendations.append("\n🍽️ DIETARY TIPS:")
    if data["stab_glu"] >= 126:
        recommendations.append("• Reduce refined sugars and simple carbohydrates")
    if data["chol"] >= 200:
        recommendations.append("• Reduce saturated fats and increase omega-3 intake")
    if data["hdl"] < 40:
        recommendations.append("• Increase intake of healthy fats (nuts, avocado, fish)")
    
    recommendations.append("\n💪 EXERCISE RECOMMENDATIONS:")
    bmi = (data["weight"] * 703) / (data["height"] ** 2)
    if bmi >= 30:
        recommendations.append("• Focus on low-impact cardio (walking, swimming)")
    recommendations.append("• Aim for 150 minutes of moderate activity per week")
    recommendations.append("• Include strength training 2-3 times per week")
    
    return "\n".join(recommendations)

def get_notifications(user_id, unread_only=False):
    """Get user notifications."""
    conn = get_db()
    cursor = conn.cursor()
    
    if unread_only:
        rows = cursor.execute(
            """SELECT * FROM notifications WHERE user_id=? AND is_read=0 
               ORDER BY created_at DESC""",
            (user_id,)
        ).fetchall()
    else:
        rows = cursor.execute(
            """SELECT * FROM notifications WHERE user_id=? 
               ORDER BY created_at DESC LIMIT 50""",
            (user_id,)
        ).fetchall()
    
    conn.close()
    return [dict(r) for r in rows]

def mark_notification_read(notification_id):
    """Mark notification as read."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE notifications SET is_read=1 WHERE id=?", (notification_id,))
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

def get_trend_data(user_id, role):
    """Get trend data for graphs."""
    conn = get_db()
    cursor = conn.cursor()
    
    if role in ("admin", "doctor"):
        rows = cursor.execute("""
            SELECT created_at, risk_score, prediction
            FROM predictions
            ORDER BY created_at ASC
        """).fetchall()
    else:
        rows = cursor.execute("""
            SELECT created_at, risk_score, prediction
            FROM predictions
            WHERE user_id=?
            ORDER BY created_at ASC
        """, (user_id,)).fetchall()
    
    conn.close()
    return [dict(r) for r in rows]

# ─── Precautions & Recommendations ────────────────────────────────────────

def get_dietary_recommendations(data):
    """Get dietary recommendations based on health metrics."""
    recommendations = {
        "foods_to_eat": [],
        "foods_to_avoid": [],
        "general_tips": []
    }
    
    # High glucose recommendations
    if data["stab_glu"] >= 126:
        recommendations["foods_to_avoid"].append("🚫 Refined sugars & simple carbs (soda, candy, white bread)")
        recommendations["foods_to_eat"].append("✅ Whole grains & high-fiber foods (oats, brown rice, vegetables)")
    
    # High cholesterol recommendations
    if data["chol"] >= 200:
        recommendations["foods_to_avoid"].append("🚫 Saturated fats & trans fats (butter, fatty meats, fried foods)")
        recommendations["foods_to_eat"].append("✅ Lean proteins, fish rich in omega-3, olive oil")
    
    # Low HDL recommendations
    if data["hdl"] < 40:
        recommendations["foods_to_eat"].append("✅ Foods with healthy fats (nuts, avocado, fatty fish)")
        recommendations["general_tips"].append("💪 Regular aerobic exercise 30-45 minutes daily")
    
    # High BMI recommendations
    bmi = (data["weight"] * 703) / (data["height"] ** 2)
    if bmi >= 30:
        recommendations["foods_to_avoid"].append("🚫 Processed foods & sugary beverages")
        recommendations["general_tips"].append("⚖️ Aim for gradual weight loss (1-2 lbs/week)")
        recommendations["general_tips"].append("🥗 Increase vegetable intake to 50% of plate")
    
    # High blood pressure
    if data["bp1s"] >= 140:
        recommendations["foods_to_avoid"].append("🚫 High sodium foods (processed foods, salty snacks)")
        recommendations["foods_to_eat"].append("✅ Potassium-rich foods (bananas, spinach, sweet potatoes)")
        recommendations["general_tips"].append("🧂 Limit salt intake to <2,300mg/day")
    
    # General recommendations
    recommendations["general_tips"].extend([
        "💧 Drink plenty of water (8-10 glasses daily)",
        "🍽️ Eat smaller, frequent meals to maintain stable blood sugar",
        "🚶 Walk after meals to help control glucose levels",
        "😴 Get 7-9 hours of quality sleep",
        "🧘 Manage stress through meditation or yoga"
    ])
    
    return recommendations

def get_lifestyle_recommendations(data, risk_score):
    """Get lifestyle recommendations based on risk level."""
    recommendations = []
    
    if risk_score >= 80:
        recommendations.extend([
            "🚨 **IMMEDIATE ACTIONS:**",
            "📋 Schedule doctor appointment immediately",
            "🩸 Get comprehensive blood work & glucose tolerance test",
            "🏥 Consider endocrinologist referral",
            "📱 Monitor blood glucose daily if advised",
            "💊 Discuss medication options with healthcare provider"
        ])
    elif risk_score >= 50:
        recommendations.extend([
            "⚠️ **RECOMMENDED ACTIONS:**",
            "📅 Schedule doctor checkup within 2 weeks",
            "🧪 Get HbA1c test (3-month average glucose)",
            "📊 Monthly glucose monitoring at home",
            "🎯 Work with nutritionist on meal planning",
            "💪 Start structured exercise program (150 min/week)"
        ])
    else:
        recommendations.extend([
            "✅ **MAINTENANCE ACTIONS:**",
            "🔄 Annual health checkups",
            "🏃 Maintain regular physical activity",
            "🥗 Continue healthy eating habits",
            "⏱️ Monitor weight quarterly",
            "👨‍⚕️ Preventive screening as recommended by doctor"
        ])
    
    return recommendations

def get_exercise_recommendations(data):
    """Get personalized exercise recommendations."""
    recommendations = []
    bmi = (data["weight"] * 703) / (data["height"] ** 2)
    
    if bmi >= 35:
        recommendations.append("🚴 Low-impact exercises: Swimming, cycling, water aerobics (start 20-30 min)")
        recommendations.append("🧘 Gentle yoga or Pilates for flexibility")
    elif bmi >= 30:
        recommendations.append("🏃 Moderate cardio: Brisk walking, jogging (150 min/week)")
        recommendations.append("🏋️ Strength training 2-3 times/week")
    else:
        recommendations.append("🏃 Maintain cardio: 150-300 min/week moderate intensity")
        recommendations.append("🏋️ Strength training 2+ times/week")
    
    recommendations.extend([
        "🚶 Walk 5-10 minutes after each meal",
        "⏰ Avoid prolonged sitting (stand/move every hour)"
    ])
    
    return recommendations

def display_precautions(data, risk_score):
    """Display comprehensive precautions and recommendations."""
    
    st.markdown("---")
    st.subheader("📋 Personalized Precautions & Recommendations")
    
    # Dietary Recommendations
    with st.expander("🍽️ **Dietary Recommendations**", expanded=True):
        diet_rec = get_dietary_recommendations(data)
        
        if diet_rec["foods_to_eat"]:
            st.write("**✅ Foods to Eat:**")
            for food in diet_rec["foods_to_eat"]:
                st.write(food)
        
        if diet_rec["foods_to_avoid"]:
            st.write("**🚫 Foods to Avoid:**")
            for food in diet_rec["foods_to_avoid"]:
                st.write(food)
        
        st.write("**💡 General Tips:**")
        for tip in diet_rec["general_tips"]:
            st.write(tip)
    
    # Lifestyle & Medical Recommendations
    with st.expander("🏥 **Lifestyle & Medical Recommendations**", expanded=True):
        lifestyle_rec = get_lifestyle_recommendations(data, risk_score)
        for rec in lifestyle_rec:
            st.write(rec)
    
    # Exercise Plan
    with st.expander("💪 **Exercise Plan**", expanded=False):
        exercise_rec = get_exercise_recommendations(data)
        st.write("**Recommended Activities:**")
        for ex in exercise_rec:
            st.write(ex)
    
    # Monitoring Schedule
    with st.expander("📊 **Monitoring & Follow-up Schedule**", expanded=False):
        if risk_score >= 80:
            st.write("**🚨 High Risk - Intensive Monitoring:**")
            st.write("• Blood glucose: Daily")
            st.write("• Doctor visits: Every 2-4 weeks")
            st.write("• Lab work (HbA1c): Every 3 months")
            st.write("• Blood pressure: 2-3 times/week")
        elif risk_score >= 50:
            st.write("**⚠️ Moderate Risk - Regular Monitoring:**")
            st.write("• Blood glucose: 2-3 times/week")
            st.write("• Doctor visits: Every 3 months")
            st.write("• Lab work (HbA1c): Every 6 months")
            st.write("• Blood pressure: Weekly")
        else:
            st.write("**✅ Low Risk - Standard Prevention:**")
            st.write("• Blood glucose: Monthly")
            st.write("• Doctor visits: Annual")
            st.write("• Lab work (HbA1c): Annual")
            st.write("• Blood pressure: Quarterly")
    
    # Warning Signs
    with st.expander("🚨 **Warning Signs - When to Seek Immediate Medical Help**", expanded=False):
        st.warning("""
        Contact emergency services immediately if you experience:
        • Sudden extreme thirst or frequent urination
        • Unexplained weight loss (5+ lbs in weeks)
        • Extreme fatigue or weakness
        • Blurred vision or difficulty concentrating
        • Fruity-smelling breath
        • Nausea, vomiting, or abdominal pain
        • Chest pain or shortness of breath
        • Severe headache or dizziness
        """)

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

# ─── Main App ───────────────────────────────────────────────────────────────

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
        role = st.selectbox("Role", ["patient", "doctor", "admin"], key="reg_role")
        
        if st.button("Create Account"):
            success, message = register(new_username, new_password, full_name, email, role)
            if success:
                st.success(message)
            else:
                st.error(message)

else:
    # Main Application
    st.title("🩺 Diabetes Prediction System")
    
    # Display unread notifications badge
    unread_notifications = get_notifications(st.session_state.user_id, unread_only=True)
    notification_badge = f" ({len(unread_notifications)})" if unread_notifications else ""
    
    # Sidebar
    with st.sidebar:
        st.write(f"**User:** {st.session_state.full_name}")
        st.write(f"**Role:** {st.session_state.role.capitalize()}")
        
        # Build navigation options based on role
        nav_options = ["Dashboard", "History", "Statistics", "Notifications", "Logout"]
        
        # Only patients can predict
        if st.session_state.role == "patient":
            nav_options.insert(1, "Predict")
        
        page = st.radio("Navigation", nav_options)
    
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
        st.write(f"Hi, {st.session_state.full_name}!")
        
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
        
        # Show role-based welcome message
        st.markdown("---")
        if st.session_state.role == "admin":
            st.info("👨‍💼 **Admin Dashboard**: View system statistics and manage users")
        elif st.session_state.role == "doctor":
            st.info("👨‍⚕️ **Doctor Dashboard**: Monitor patient predictions and provide recommendations")
        else:
            st.info("👤 **Patient Dashboard**: Track your health and manage predictions")
    
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
            
            # Count flags and track which ones
            flags = 0
            flagged_features = []
            
            if stab_glu >= 126:
                flags += 1
                flagged_features.append(f"🔴 High Fasting Glucose: {stab_glu} mg/dL (≥ 126)")
            if ratio >= 7:
                flags += 1
                flagged_features.append(f"🔴 High Chol/HDL Ratio: {ratio:.1f} (≥ 7)")
            if hdl < 40:
                flags += 1
                flagged_features.append(f"🔴 Low HDL Cholesterol: {hdl} mg/dL (< 40)")
            bmi = (weight * 703) / (height ** 2)
            if bmi >= 30:
                flags += 1
                flagged_features.append(f"🔴 High BMI: {bmi:.1f} (≥ 30)")
            if bp1s >= 140:
                flags += 1
                flagged_features.append(f"🔴 High Systolic BP: {bp1s} mmHg (≥ 140)")
            if waist >= 40:
                flags += 1
                flagged_features.append(f"🔴 High Waist Circumference: {waist} inches (≥ 40)")
            if age >= 60:
                flags += 1
                flagged_features.append(f"🔴 Age ≥ 60: {int(age)} years")
            
            # Save to database
            pred_id = save_prediction(st.session_state.user_id, data, prob, prediction, flags)
            
            # Create notification
            create_notification(st.session_state.user_id, pred_id, risk_score, prediction, data)
            
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
                st.metric("BMI", f"{bmi:.1f}")
                st.metric("High-Risk Flags", flags)
                st.write(f"**WHR:** {waist/hip:.2f}")
            
            # Display flagged features
            if flagged_features:
                st.subheader("🚩 Flagged Risk Factors")
                for feature in flagged_features:
                    st.warning(feature)
            else:
                st.success("✅ No high-risk factors detected!")
            
            # Risk assessment
            if risk_score >= 80:
                st.warning("⚠️ **High Risk** - Immediate medical consultation recommended")
            elif risk_score >= 50:
                st.info("⚠️ **Moderate Risk** - Schedule a checkup with your doctor")
            else:
                st.success("✅ **Low Risk** - Continue healthy lifestyle habits")
            
            # Display precautions and recommendations
            display_precautions(data, risk_score)
            
            # Show notification info
            st.info("📬 A detailed notification has been sent to your account. Check the Notifications page.")
    
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
        else:
            st.info("No prediction history available.")
    
    elif page == "Statistics":
        st.subheader("Statistics & Analytics")
        
        stats = get_stats(st.session_state.user_id, st.session_state.role)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", stats.get("total", 0))
        with col2:
            st.metric("Diabetic Cases", stats.get("diabetic", 0))
        with col3:
            st.metric("Non-Diabetic Cases", stats.get("non_diabetic", 0))
        with col4:
            st.metric("Avg Risk Score", f"{stats.get('avg_risk', 0):.1f}%")
        
        st.markdown("---")
        
        # Get trend data
        trend_data = get_trend_data(st.session_state.user_id, st.session_state.role)
        
        if trend_data:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(trend_data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Calculate percentages
            total = stats.get("total", 1)
            diabetic = stats.get("diabetic", 0)
            non_diabetic = stats.get("non_diabetic", 0)
            diabetic_pct = (diabetic / total * 100) if total > 0 else 0
            non_diabetic_pct = 100 - diabetic_pct
            
            # Risk distribution pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Prediction Distribution")
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Diabetic', 'Non-Diabetic'],
                    values=[diabetic, non_diabetic],
                    marker=dict(colors=['#FF6B6B', '#51CF66']),
                    textposition='inside',
                    textinfo='label+percent'
                )])
                fig_pie.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("📈 Risk Score Distribution")
                fig_hist = go.Figure(data=[go.Histogram(
                    x=df['risk_score'],
                    nbinsx=15,
                    marker=dict(color='#4C72B0')
                )])
                fig_hist.update_xaxes(title_text="Risk Score (%)")
                fig_hist.update_yaxes(title_text="Frequency")
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Risk score trend over time
            st.subheader("📈 Risk Score Trend Over Time")
            df_sorted = df.sort_values('created_at')
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=df_sorted['created_at'],
                y=df_sorted['risk_score'],
                mode='lines+markers',
                name='Risk Score',
                line=dict(color='#FF9999', width=3),
                marker=dict(size=8)
            ))
            
            fig_trend.add_hline(y=50, line_dash="dash", line_color="orange", 
                               annotation_text="Moderate Risk Threshold (50%)")
            fig_trend.add_hline(y=80, line_dash="dash", line_color="red",
                               annotation_text="High Risk Threshold (80%)")
            
            fig_trend.update_xaxes(title_text="Date")
            fig_trend.update_yaxes(title_text="Risk Score (%)")
            fig_trend.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Risk category breakdown
            st.subheader("🎯 Risk Category Breakdown")
            high_risk = len(df[df['risk_score'] >= 80])
            moderate_risk = len(df[(df['risk_score'] >= 50) & (df['risk_score'] < 80)])
            low_risk = len(df[df['risk_score'] < 50])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 High Risk (≥80%)", high_risk)
            with col2:
                st.metric("⚠️ Moderate Risk (50-80%)", moderate_risk)
            with col3:
                st.metric("✅ Low Risk (<50%)", low_risk)
            
            # Bar chart for risk categories
            fig_bar = go.Figure(data=[
                go.Bar(x=['Low Risk', 'Moderate Risk', 'High Risk'],
                       y=[low_risk, moderate_risk, high_risk],
                       marker=dict(color=['#51CF66', '#FFD93D', '#FF6B6B']))
            ])
            fig_bar.update_yaxes(title_text="Number of Cases")
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        else:
            st.info("No prediction data available for analysis.")
    
    elif page == "Notifications":
        st.subheader("📬 Notifications")
        
        notifications = get_notifications(st.session_state.user_id)
        
        if notifications:
            # Filter tabs
            col1, col2 = st.columns(2)
            with col1:
                unread_count = len([n for n in notifications if not n['is_read']])
                st.metric("Unread Notifications", unread_count)
            
            for notif in notifications:
                with st.container():
                    st.markdown("---")
                    
                    # Header with title and date
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{notif['title']}**")
                        st.caption(f"📅 {notif['created_at']}")
                    with col2:
                        status = "✅ Read" if notif['is_read'] else "🆕 New"
                        st.caption(status)
                    
                    # Message
                    st.write(notif['message'])
                    
                    # Recommendations
                    if notif['recommendations']:
                        with st.expander("📋 View Recommendations"):
                            st.text(notif['recommendations'])
                    
                    # Mark as read button
                    if not notif['is_read']:
                        if st.button("Mark as Read", key=f"notif_{notif['id']}"):
                            mark_notification_read(notif['id'])
                            st.success("Notification marked as read!")
                            st.rerun()
        else:
            st.info("📭 No notifications available.")
