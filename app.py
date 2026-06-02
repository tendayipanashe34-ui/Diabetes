"""
Diabetes Prediction — Full Stack Application
Backend  : Flask + SQLite
Auth     : Session-based (username + hashed password)
Database : users, predictions tables
Model    : Deep ANN logistic scoring (mirrors notebook)
"""

import sqlite3, hashlib, os, json, math
from datetime import datetime
from functools import wraps
from flask import ( # type: ignore
    Flask, request, jsonify, session,
    render_template, redirect, url_for, g
)

app = Flask(__name__)
app.secret_key = os.urandom(24)          # change to fixed key in production

DB_PATH = os.path.join(os.path.dirname(__file__), "diabetes.db")

# ─── Database helpers ─────────────────────────────────────────────────────────

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, "_database", None)
    if db: db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
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
                -- patient demographics
                gender        TEXT,
                -- metabolic
                chol          REAL,   chol_label      TEXT,
                stab_glu      REAL,   stab_glu_label  TEXT,
                hdl           REAL,   hdl_label       TEXT,
                ratio         REAL,   ratio_label     TEXT,
                -- physical
                age           REAL,   age_label       TEXT,
                height        REAL,   height_label    TEXT,
                weight        REAL,   weight_label    TEXT,
                waist         REAL,   waist_label     TEXT,
                hip           REAL,   hip_label       TEXT,
                -- bp & timing
                bp1s          REAL,   bp1s_label      TEXT,
                bp1d          REAL,   bp1d_label      TEXT,
                time_ppn      REAL,   time_ppn_label  TEXT,
                -- result
                risk_score    REAL,
                probability   REAL,
                prediction    TEXT,
                flagged_count INTEGER,
                created_at    TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)
        db.commit()
        # seed demo accounts
        _seed_users(db)

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def _seed_users(db):
    seeds = [
        ("admin",   "admin123",   "Administrator",  "admin@clinic.com",  "admin"),
        ("doctor1", "doctor123",  "Dr. Jane Smith", "jsmith@clinic.com", "doctor"),
        ("patient1","patient123", "John Doe",       "jdoe@clinic.com",   "patient"),
    ]
    for uname, pw, name, email, role in seeds:
        try:
            db.execute(
                "INSERT INTO users (username,password_hash,full_name,email,role) VALUES (?,?,?,?,?)",
                (uname, _hash(pw), name, email, role)
            )
        except sqlite3.IntegrityError:
            pass
    db.commit()

# ─── Auth decorator ───────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return decorated

# ─── Prediction model (mirrors notebook Deep ANN) ────────────────────────────

def _zscore(v, mean, std): return (v - mean) / std

def _sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

SCALER = {             # approximate StandardScaler params from dataset
    "stab_glu": (107, 45),  "ratio":  (4.5, 1.5), "hdl":   (50,  17),
    "age":      (46,  16),  "chol":   (207, 45),  "bp1s":  (136, 23),
    "bp1d":     (83,  14),  "weight": (177, 44),  "waist": (36,  6),
    "time_ppn": (90,  60),  "height": (66,  4),   "hip":   (40,  5),
}

def compute_probability(v, gender):
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

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("index"))
    return render_template("index.html")

# ── Auth API ──

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    username  = (data.get("username") or "").strip()
    password  = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()
    email     = (data.get("email") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username,password_hash,full_name,email) VALUES (?,?,?,?)",
            (username, _hash(password), full_name, email)
        )
        db.commit()
        return jsonify({"message": "Account created successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 409

@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    db       = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, _hash(password))
    ).fetchone()
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401
    session["user_id"]   = user["id"]
    session["username"]  = user["username"]
    session["full_name"] = user["full_name"]
    session["role"]      = user["role"]
    return jsonify({
        "message":   "Login successful",
        "username":  user["username"],
        "full_name": user["full_name"],
        "role":      user["role"],
    })

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})

@app.route("/api/me")
def me():
    if "user_id" not in session:
        return jsonify({"authenticated": False})
    return jsonify({
        "authenticated": True,
        "user_id":   session["user_id"],
        "username":  session["username"],
        "full_name": session["full_name"],
        "role":      session["role"],
    })

# ── Prediction API ──

@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json()
    required = ["gender","chol","stab_glu","hdl","ratio","age",
                "height","weight","waist","hip","bp1s","bp1d","time_ppn"]
    for k in required:
        if k not in data:
            return jsonify({"error": f"Missing field: {k}"}), 400

    gender = data["gender"]
    nums   = {k: float(data[k]) for k in required if k != "gender"}

    prob       = compute_probability(nums, gender)
    risk_score = round(prob * 100, 1)
    prediction = "Diabetic" if prob >= 0.5 else "Non-Diabetic"

    # Count high-risk flags
    flags = 0
    if nums["stab_glu"] >= 126: flags += 1
    if nums["ratio"]    >= 7:   flags += 1
    if nums["hdl"]      < 40:   flags += 1
    bmi = (nums["weight"] * 703) / (nums["height"] ** 2)
    if bmi >= 30:               flags += 1
    if nums["bp1s"]     >= 140: flags += 1
    if nums["waist"]    >= 40:  flags += 1
    if nums["age"]      >= 60:  flags += 1

    # Persist to DB
    db = get_db()
    db.execute(
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
            session["user_id"], gender,
            nums["chol"],     data.get("chol_label",""),
            nums["stab_glu"], data.get("stab_glu_label",""),
            nums["hdl"],      data.get("hdl_label",""),
            nums["ratio"],    data.get("ratio_label",""),
            nums["age"],      data.get("age_label",""),
            nums["height"],   data.get("height_label",""),
            nums["weight"],   data.get("weight_label",""),
            nums["waist"],    data.get("waist_label",""),
            nums["hip"],      data.get("hip_label",""),
            nums["bp1s"],     data.get("bp1s_label",""),
            nums["bp1d"],     data.get("bp1d_label",""),
            nums["time_ppn"], data.get("time_ppn_label",""),
            risk_score, round(prob, 4), prediction, flags,
        )
    )
    db.commit()
    pred_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    return jsonify({
        "prediction_id": pred_id,
        "prediction":    prediction,
        "probability":   round(prob * 100, 1),
        "risk_score":    risk_score,
        "flagged_count": flags,
        "bmi":           round(bmi, 1),
    })

# ── History API ──

@app.route("/api/history")
@login_required
def history():
    db   = get_db()
    role = session.get("role")
    uid  = session["user_id"]
    if role in ("admin", "doctor"):
        rows = db.execute("""
            SELECT p.*, u.username, u.full_name
            FROM predictions p JOIN users u ON p.user_id=u.id
            ORDER BY p.created_at DESC LIMIT 100
        """).fetchall()
    else:
        rows = db.execute("""
            SELECT p.*, u.username, u.full_name
            FROM predictions p JOIN users u ON p.user_id=u.id
            WHERE p.user_id=?
            ORDER BY p.created_at DESC LIMIT 50
        """, (uid,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/stats")
@login_required
def stats():
    db  = get_db()
    uid = session["user_id"]
    r   = session.get("role")
    scope = "" if r in ("admin","doctor") else f"WHERE user_id={uid}"
    row = db.execute(f"""
        SELECT
          COUNT(*) total,
          SUM(CASE WHEN prediction='Diabetic' THEN 1 ELSE 0 END) diabetic,
          SUM(CASE WHEN prediction='Non-Diabetic' THEN 1 ELSE 0 END) non_diabetic,
          ROUND(AVG(risk_score),1) avg_risk
        FROM predictions {scope}
    """).fetchone()
    return jsonify(dict(row))

@app.route("/reports")
@login_required
def reports():
    try:
        with open('classification_reports.json', 'r') as f:
            reports_data = json.load(f)
        return render_template("reports.html", reports=reports_data)
    except FileNotFoundError:
        return jsonify({"error": "Reports not available"}), 404
