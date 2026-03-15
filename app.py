from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
import pandas as pd
import sqlite3
import joblib
import os
import random
from datetime import datetime
from functools import wraps
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "fraud_secret_key_123"

DB_NAME = "fraud_system.db"
MODEL_PATH = "fraud_pipeline.joblib"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
feature_cols = artifacts["feature_cols"]
median_purchase_value = artifacts["median_purchase_value"]
model_name = artifacts.get("model_name", "RandomForest_v2")
model_trained_at = artifacts.get("trained_at", "")


# -----------------------------
# DB helpers
# -----------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


# -----------------------------
# Auth helpers
# -----------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def role_required(required_role):
    def wrapper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user" not in session:
                return redirect(url_for("login"))
            if session.get("role") != required_role:
                flash("Access denied", "danger")
                return redirect(url_for("dashboard"))
            return f(*args, **kwargs)
        return decorated
    return wrapper


# -----------------------------
# Notification helpers
# -----------------------------
def create_notification(title, message, severity="info", conn=None):
    own_conn = False
    if conn is None:
        conn = get_db_connection()
        own_conn = True

    conn.execute("""
        INSERT INTO notifications (title, message, severity, is_read, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        title,
        message,
        severity,
        0,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    if own_conn:
        conn.commit()
        conn.close()


# -----------------------------
# Simple settings
# -----------------------------
def get_settings():
    return {
        "review_threshold": 0.40,
        "otp_threshold": 0.65,
        "block_threshold": 0.85,
        "high_purchase_multiplier": 2.0,
        "new_account_hours": 24.0,
        "duplicate_hours_window": 2.0
    }


# -----------------------------
# Feature engineering
# -----------------------------
def add_engineered_features(df):
    df = df.copy()

    if "customer_ref" not in df.columns:
        df["customer_ref"] = ""

    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
    df = df.dropna(subset=["signup_time", "purchase_time"])

    df["purchase_value"] = pd.to_numeric(df["purchase_value"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["purchase_value", "age"])

    df["account_age_hours"] = (
        (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600
    )
    df["purchase_hour"] = df["purchase_time"].dt.hour
    df["purchase_day"] = df["purchase_time"].dt.day
    df["purchase_month"] = df["purchase_time"].dt.month
    df["purchase_weekday"] = df["purchase_time"].dt.weekday
    df["is_night_transaction"] = df["purchase_hour"].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
    df["is_new_account"] = df["account_age_hours"].apply(lambda x: 1 if x < 24 else 0)
    df["high_value_transaction"] = df["purchase_value"].apply(lambda x: 1 if x > median_purchase_value else 0)

    df = df[df["purchase_value"] >= 0]
    df = df[df["age"] >= 0]
    df = df[df["account_age_hours"] >= 0]

    df["age"] = df["age"].astype(int)
    df["customer_ref"] = df["customer_ref"].astype(str).fillna("").str.strip()

    return df


# -----------------------------
# Fraud logic
# -----------------------------
def detect_duplicate(customer_ref, purchase_value, purchase_time, settings):
    if not customer_ref:
        return 0

    with get_db_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS cnt
            FROM transactions
            WHERE lower(customer_ref) = lower(?)
              AND purchase_value = ?
              AND ABS((julianday(purchase_time) - julianday(?)) * 24) <= ?
        """, (
            customer_ref,
            purchase_value,
            purchase_time,
            settings["duplicate_hours_window"]
        )).fetchone()

    return 1 if row["cnt"] > 0 else 0


def rule_engine(row, settings, duplicate_flag):
    rule_score = 0.0
    reasons = []

    if row["account_age_hours"] < 1:
        rule_score += 0.25
        reasons.append("Very new account")

    if row["purchase_value"] > median_purchase_value * settings["high_purchase_multiplier"]:
        rule_score += 0.20
        reasons.append("Unusually high purchase value")

    if row["is_night_transaction"] == 1:
        rule_score += 0.15
        reasons.append("Night transaction")

    if row["account_age_hours"] < settings["new_account_hours"]:
        rule_score += 0.20
        reasons.append("Account created recently")

    if row["age"] < 21 and row["purchase_value"] > median_purchase_value:
        rule_score += 0.10
        reasons.append("Young user with high-value transaction")

    if duplicate_flag == 1:
        rule_score += 0.20
        reasons.append("Possible duplicate transaction")

    rule_score = min(rule_score, 0.95)
    return rule_score, reasons


def combine_scores(ml_prob, rule_score):
    final_prob = (0.75 * ml_prob) + (0.25 * rule_score)
    return min(max(final_prob, 0.0), 1.0)


def get_risk_level(prob, settings):
    if prob >= settings["block_threshold"]:
        return "Very High"
    if prob >= settings["otp_threshold"]:
        return "High"
    if prob >= settings["review_threshold"]:
        return "Medium"
    return "Low"


def get_action(prob, settings):
    if prob >= settings["block_threshold"]:
        return "Block"
    if prob >= settings["otp_threshold"]:
        return "OTP Verification"
    if prob >= settings["review_threshold"]:
        return "Review"
    return "Allow"


def get_confidence_band(prob):
    if prob >= 0.85 or prob <= 0.15:
        return "High Confidence"
    if prob >= 0.65 or prob <= 0.35:
        return "Medium Confidence"
    return "Low Confidence"


def generate_otp():
    return str(random.randint(100000, 999999))


def create_otp(transaction_id, conn=None):
    otp_code = generate_otp()
    own_conn = False
    if conn is None:
        conn = get_db_connection()
        own_conn = True

    conn.execute("""
        INSERT INTO otp_verifications (transaction_id, otp_code, status, generated_at, verified_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        transaction_id,
        otp_code,
        "Pending",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        None
    ))

    create_notification(
        "OTP Required",
        f"OTP verification required for transaction ID {transaction_id}",
        "warning",
        conn=conn
    )

    if own_conn:
        conn.commit()
        conn.close()

    return otp_code


def assign_case_to_analyst(conn=None):
    own_conn = False
    if conn is None:
        conn = get_db_connection()
        own_conn = True

    analysts = conn.execute(
        "SELECT username FROM users WHERE role = 'analyst' ORDER BY username"
    ).fetchall()

    if not analysts:
        if own_conn:
            conn.close()
        return "unassigned"

    best_analyst = None
    min_open_cases = None

    for analyst in analysts:
        username = analyst["username"]
        open_cases = conn.execute("""
            SELECT COUNT(*) AS cnt
            FROM cases
            WHERE assigned_to = ? AND status IN ('Pending', 'Under Review')
        """, (username,)).fetchone()["cnt"]

        if min_open_cases is None or open_cases < min_open_cases:
            min_open_cases = open_cases
            best_analyst = username

    if own_conn:
        conn.close()

    return best_analyst or "unassigned"


def save_transaction(row_dict, created_by, source_type, conn=None):
    own_conn = False
    if conn is None:
        conn = get_db_connection()
        own_conn = True

    cur = conn.cursor()

    cur.execute("""
        INSERT INTO transactions (
            customer_ref, purchase_value, age, signup_time, purchase_time, account_age_hours,
            purchase_hour, ml_probability, rule_score, fraud_probability, confidence_band,
            duplicate_flag, risk_level, action_recommendation, otp_status, final_prediction,
            suspicious_reasons, source_type, created_by, feedback_label, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row_dict["customer_ref"],
        row_dict["purchase_value"],
        row_dict["age"],
        row_dict["signup_time"],
        row_dict["purchase_time"],
        row_dict["account_age_hours"],
        row_dict["purchase_hour"],
        row_dict["ml_probability"],
        row_dict["rule_score"],
        row_dict["fraud_probability"],
        row_dict["confidence_band"],
        row_dict["duplicate_flag"],
        row_dict["risk_level"],
        row_dict["action_recommendation"],
        row_dict["otp_status"],
        row_dict["final_prediction"],
        row_dict["suspicious_reasons"],
        source_type,
        created_by,
        None,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    transaction_id = cur.lastrowid

    if row_dict["action_recommendation"] in ["Review", "Block", "OTP Verification"]:
        assigned_to = assign_case_to_analyst(conn=conn)
        cur.execute("""
            INSERT INTO cases (transaction_id, status, assigned_to, notes, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            transaction_id,
            "Pending",
            assigned_to,
            "Auto-created from suspicious prediction",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        create_notification(
            "New Case Assigned",
            f"Case created for transaction ID {transaction_id} and assigned to {assigned_to}",
            "info",
            conn=conn
        )

    if own_conn:
        conn.commit()
        conn.close()

    return transaction_id


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db_connection() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS cnt FROM transactions").fetchone()["cnt"]
        fraud = conn.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE final_prediction = 1").fetchone()["cnt"]
        review = conn.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE action_recommendation = 'Review'").fetchone()["cnt"]
        blocked = conn.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE action_recommendation = 'Block'").fetchone()["cnt"]
        otp_pending = conn.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE action_recommendation = 'OTP Verification' AND otp_status = 'Pending'").fetchone()["cnt"]
        duplicates = conn.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE duplicate_flag = 1").fetchone()["cnt"]
        loss_prevented = conn.execute("SELECT COALESCE(SUM(purchase_value), 0) AS amt FROM transactions WHERE action_recommendation = 'Block'").fetchone()["amt"]

        recent = conn.execute("""
            SELECT id, customer_ref, purchase_value, fraud_probability, confidence_band, risk_level,
                   action_recommendation, otp_status, duplicate_flag, created_at
            FROM transactions
            ORDER BY id DESC LIMIT 10
        """).fetchall()

        notif_count = conn.execute("SELECT COUNT(*) AS cnt FROM notifications WHERE is_read = 0").fetchone()["cnt"]

    fraud_rate = round((fraud / total) * 100, 2) if total else 0

    return render_template(
        "dashboard.html",
        total=total,
        fraud=fraud,
        review=review,
        blocked=blocked,
        otp_pending=otp_pending,
        duplicates=duplicates,
        fraud_rate=fraud_rate,
        loss_prevented=round(loss_prevented, 2),
        recent=recent,
        model_name=model_name,
        model_trained_at=model_trained_at,
        notif_count=notif_count
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None

    if request.method == "POST":
        try:
            purchase_value = float(request.form["purchase_value"])
            age = int(request.form["age"])
            signup_time = request.form["signup_time"]
            purchase_time = request.form["purchase_time"]
            customer_ref = request.form.get("customer_ref", "").strip()

            input_df = pd.DataFrame({
                "customer_ref": [customer_ref],
                "purchase_value": [purchase_value],
                "age": [age],
                "signup_time": [signup_time],
                "purchase_time": [purchase_time]
            })

            processed = add_engineered_features(input_df)
            if processed.empty:
                flash("Invalid input values", "danger")
                return redirect(url_for("predict"))

            settings = get_settings()
            X = processed[feature_cols]
            ml_prob = float(model.predict_proba(X)[0][1])

            row = processed.iloc[0].to_dict()

            duplicate_flag = detect_duplicate(
                customer_ref,
                row["purchase_value"],
                str(pd.to_datetime(row["purchase_time"])),
                settings
            )

            rule_score, reasons = rule_engine(row, settings, duplicate_flag)
            final_prob = combine_scores(ml_prob, rule_score)
            confidence_band = get_confidence_band(final_prob)
            risk_level = get_risk_level(final_prob, settings)
            action = get_action(final_prob, settings)
            final_pred = 1 if final_prob >= 0.50 else 0

            suspicious_reasons = ", ".join(reasons) if reasons else "No major rule triggers"
            otp_status = "Not Required"

            row_dict = {
                "customer_ref": customer_ref,
                "purchase_value": row["purchase_value"],
                "age": int(row["age"]),
                "signup_time": str(pd.to_datetime(row["signup_time"])),
                "purchase_time": str(pd.to_datetime(row["purchase_time"])),
                "account_age_hours": round(float(row["account_age_hours"]), 2),
                "purchase_hour": int(row["purchase_hour"]),
                "ml_probability": round(ml_prob * 100, 2),
                "rule_score": round(rule_score * 100, 2),
                "fraud_probability": round(final_prob * 100, 2),
                "confidence_band": confidence_band,
                "duplicate_flag": duplicate_flag,
                "risk_level": risk_level,
                "action_recommendation": action,
                "otp_status": otp_status,
                "final_prediction": final_pred,
                "suspicious_reasons": suspicious_reasons
            }

            with get_db_connection() as conn:
                transaction_id = save_transaction(row_dict, session["user"], "manual", conn=conn)

                otp_code = None
                if action == "OTP Verification":
                    otp_code = create_otp(transaction_id, conn=conn)
                    conn.execute(
                        "UPDATE transactions SET otp_status = ? WHERE id = ?",
                        ("Pending", transaction_id)
                    )

                if risk_level in ["High", "Very High"]:
                    create_notification(
                        "High Risk Transaction",
                        f"Transaction ID {transaction_id} flagged with {row_dict['fraud_probability']}% probability",
                        "danger",
                        conn=conn
                    )

                conn.commit()

            result = {
                "transaction_id": transaction_id,
                "prediction": "Fraudulent" if final_pred == 1 else "Legitimate",
                "fraud_probability": row_dict["fraud_probability"],
                "confidence_band": confidence_band,
                "risk_level": risk_level,
                "action": action,
                "account_age_hours": row_dict["account_age_hours"],
                "purchase_hour": row_dict["purchase_hour"],
                "suspicious_reasons": suspicious_reasons,
                "ml_probability": row_dict["ml_probability"],
                "rule_score": row_dict["rule_score"],
                "duplicate_flag": duplicate_flag,
                "otp_code": otp_code
            }

        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    return render_template("predict.html", result=result)


@app.route("/simulate", methods=["GET", "POST"])
@login_required
def simulate():
    sim_result = None

    if request.method == "POST":
        try:
            purchase_value = float(request.form["purchase_value"])
            age = int(request.form["age"])
            signup_time = request.form["signup_time"]
            purchase_time = request.form["purchase_time"]
            customer_ref = request.form.get("customer_ref", "").strip()

            input_df = pd.DataFrame({
                "customer_ref": [customer_ref],
                "purchase_value": [purchase_value],
                "age": [age],
                "signup_time": [signup_time],
                "purchase_time": [purchase_time]
            })

            processed = add_engineered_features(input_df)
            if processed.empty:
                flash("Invalid simulation values", "danger")
                return render_template("simulate.html", sim_result=None)

            settings = get_settings()
            X = processed[feature_cols]
            ml_prob = float(model.predict_proba(X)[0][1])
            row = processed.iloc[0].to_dict()

            duplicate_flag = 0
            rule_score, reasons = rule_engine(row, settings, duplicate_flag)
            final_prob = combine_scores(ml_prob, rule_score)
            confidence_band = get_confidence_band(final_prob)
            risk_level = get_risk_level(final_prob, settings)
            action = get_action(final_prob, settings)

            sim_result = {
                "ml_probability": round(ml_prob * 100, 2),
                "rule_score": round(rule_score * 100, 2),
                "final_probability": round(final_prob * 100, 2),
                "confidence_band": confidence_band,
                "risk_level": risk_level,
                "action": action,
                "reasons": ", ".join(reasons) if reasons else "No major rule triggers"
            }

        except Exception as e:
            flash(f"Simulation error: {e}", "danger")

    return render_template("simulate.html", sim_result=sim_result)


@app.route("/otp/<int:transaction_id>", methods=["GET", "POST"])
@login_required
def otp_verify(transaction_id):
    with get_db_connection() as conn:
        tx = conn.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,)).fetchone()
        otp_row = conn.execute("""
            SELECT * FROM otp_verifications
            WHERE transaction_id = ?
            ORDER BY id DESC LIMIT 1
        """, (transaction_id,)).fetchone()

        if not tx or not otp_row:
            flash("OTP record not found", "danger")
            return redirect(url_for("history"))

        if request.method == "POST":
            entered_otp = request.form.get("otp_code", "").strip()

            if entered_otp == otp_row["otp_code"]:
                conn.execute("""
                    UPDATE otp_verifications
                    SET status = ?, verified_at = ?
                    WHERE id = ?
                """, ("Verified", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), otp_row["id"]))

                conn.execute("""
                    UPDATE transactions
                    SET otp_status = ?, action_recommendation = ?, final_prediction = ?
                    WHERE id = ?
                """, ("Verified", "Allow", 0, transaction_id))

                create_notification(
                    "OTP Success",
                    f"OTP verified for transaction ID {transaction_id}",
                    "success",
                    conn=conn
                )
                conn.commit()
                flash("OTP verified successfully. Transaction allowed.", "success")
                return redirect(url_for("history"))
            else:
                conn.execute("""
                    UPDATE otp_verifications
                    SET status = ?
                    WHERE id = ?
                """, ("Failed", otp_row["id"]))

                conn.execute("""
                    UPDATE transactions
                    SET otp_status = ?, action_recommendation = ?, final_prediction = ?
                    WHERE id = ?
                """, ("Failed", "Block", 1, transaction_id))

                create_notification(
                    "OTP Failed",
                    f"OTP failed for transaction ID {transaction_id}",
                    "danger",
                    conn=conn
                )
                conn.commit()
                flash("OTP verification failed. Transaction blocked.", "danger")
                return redirect(url_for("history"))

    return render_template("otp_verify.html", tx=tx, otp_row=otp_row)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    results = []
    summary = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Please choose a CSV file", "danger")
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            df = normalize_columns(df)

            required_cols = ["purchase_value", "age", "signup_time", "purchase_time"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                flash(f"Missing columns: {missing}", "danger")
                return redirect(url_for("upload"))

            processed = add_engineered_features(df)
            if processed.empty:
                flash("No valid rows found after preprocessing", "danger")
                return redirect(url_for("upload"))

            settings = get_settings()
            X = processed[feature_cols]
            ml_probs = model.predict_proba(X)[:, 1]

            export_rows = []

            with get_db_connection() as conn:
                for i, (_, row) in enumerate(processed.iterrows()):
                    row = row.to_dict()
                    customer_ref = row.get("customer_ref", "")

                    duplicate_flag = detect_duplicate(
                        customer_ref,
                        row["purchase_value"],
                        str(pd.to_datetime(row["purchase_time"])),
                        settings
                    )

                    rule_score, reasons = rule_engine(row, settings, duplicate_flag)
                    final_prob = combine_scores(float(ml_probs[i]), rule_score)
                    confidence_band = get_confidence_band(final_prob)
                    risk_level = get_risk_level(final_prob, settings)
                    action = get_action(final_prob, settings)
                    final_pred = 1 if final_prob >= 0.50 else 0
                    suspicious_reasons = ", ".join(reasons) if reasons else "No major rule triggers"
                    otp_status = "Pending" if action == "OTP Verification" else "Not Required"

                    row_dict = {
                        "customer_ref": customer_ref,
                        "purchase_value": row["purchase_value"],
                        "age": int(row["age"]),
                        "signup_time": str(pd.to_datetime(row["signup_time"])),
                        "purchase_time": str(pd.to_datetime(row["purchase_time"])),
                        "account_age_hours": round(float(row["account_age_hours"]), 2),
                        "purchase_hour": int(row["purchase_hour"]),
                        "ml_probability": round(float(ml_probs[i]) * 100, 2),
                        "rule_score": round(rule_score * 100, 2),
                        "fraud_probability": round(final_prob * 100, 2),
                        "confidence_band": confidence_band,
                        "duplicate_flag": duplicate_flag,
                        "risk_level": risk_level,
                        "action_recommendation": action,
                        "otp_status": otp_status,
                        "final_prediction": final_pred,
                        "suspicious_reasons": suspicious_reasons
                    }

                    transaction_id = save_transaction(row_dict, session["user"], "csv", conn=conn)

                    if action == "OTP Verification":
                        create_otp(transaction_id, conn=conn)

                    if risk_level in ["High", "Very High"]:
                        create_notification(
                            "High Risk CSV Transaction",
                            f"CSV transaction ID {transaction_id} flagged at {row_dict['fraud_probability']}%",
                            "danger",
                            conn=conn
                        )

                    export_rows.append({
                        "transaction_id": transaction_id,
                        "customer_ref": customer_ref,
                        "purchase_value": row_dict["purchase_value"],
                        "age": row_dict["age"],
                        "account_age_hours": row_dict["account_age_hours"],
                        "purchase_hour": row_dict["purchase_hour"],
                        "fraud_probability": row_dict["fraud_probability"],
                        "confidence_band": confidence_band,
                        "risk_level": risk_level,
                        "action_recommendation": action,
                        "final_prediction": "Fraud" if final_pred == 1 else "Legitimate",
                        "suspicious_reasons": suspicious_reasons
                    })

                conn.commit()

            total = len(export_rows)
            fraud = sum(1 for r in export_rows if r["final_prediction"] == "Fraud")
            review = sum(1 for r in export_rows if r["action_recommendation"] == "Review")
            blocked = sum(1 for r in export_rows if r["action_recommendation"] == "Block")

            summary = {
                "total": total,
                "fraud": fraud,
                "review": review,
                "blocked": blocked,
                "fraud_rate": round((fraud / total) * 100, 2) if total else 0
            }
            results = export_rows

        except Exception as e:
            flash(f"Upload processing error: {e}", "danger")

    return render_template("upload.html", results=results, summary=summary)


@app.route("/history")
@login_required
def history():
    search = request.args.get("search", "").strip()
    risk = request.args.get("risk_level", "").strip()
    action = request.args.get("action", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()

    query = "SELECT * FROM transactions WHERE 1=1"
    params = []

    if search:
        query += " AND (customer_ref LIKE ? OR created_by LIKE ? OR suspicious_reasons LIKE ?)"
        like_value = f"%{search}%"
        params.extend([like_value, like_value, like_value])

    if risk:
        query += " AND risk_level = ?"
        params.append(risk)

    if action:
        query += " AND action_recommendation = ?"
        params.append(action)

    if date_from:
        query += " AND date(created_at) >= date(?)"
        params.append(date_from)

    if date_to:
        query += " AND date(created_at) <= date(?)"
        params.append(date_to)

    query += " ORDER BY id DESC LIMIT 300"

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    return render_template("history.html", rows=rows)


@app.route("/cases", methods=["GET", "POST"])
@login_required
def cases():
    with get_db_connection() as conn:
        if request.method == "POST":
            action_type = request.form.get("action_type")

            if action_type == "update_case":
                case_id = request.form.get("case_id")
                status = request.form.get("status")
                notes = request.form.get("notes", "").strip()
                assigned_to = request.form.get("assigned_to", "").strip()

                conn.execute("""
                    UPDATE cases SET status = ?, notes = ?, assigned_to = ? WHERE id = ?
                """, (status, notes, assigned_to, case_id))
                conn.commit()
                flash("Case updated successfully", "success")

            elif action_type == "add_note":
                case_id = request.form.get("case_id")
                note_text = request.form.get("note_text", "").strip()

                if note_text:
                    conn.execute("""
                        INSERT INTO case_notes (case_id, note_text, added_by, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (
                        case_id,
                        note_text,
                        session["user"],
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ))
                    conn.commit()
                    flash("Case note added", "success")

        analysts = conn.execute(
            "SELECT username FROM users WHERE role = 'analyst' ORDER BY username"
        ).fetchall()

        rows = conn.execute("""
            SELECT cases.id, cases.transaction_id, cases.status, cases.assigned_to, cases.notes,
                   cases.created_at, transactions.customer_ref, transactions.fraud_probability,
                   transactions.risk_level, transactions.action_recommendation
            FROM cases
            LEFT JOIN transactions ON cases.transaction_id = transactions.id
            ORDER BY cases.id DESC
        """).fetchall()

    return render_template("cases.html", rows=rows, analysts=analysts)


@app.route("/case_timeline/<int:case_id>")
@login_required
def case_timeline(case_id):
    with get_db_connection() as conn:
        case_row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        notes = conn.execute("""
            SELECT * FROM case_notes WHERE case_id = ? ORDER BY id DESC
        """, (case_id,)).fetchall()

    return render_template("case_timeline.html", case_row=case_row, notes=notes)


@app.route("/feedback/<int:transaction_id>", methods=["POST"])
@login_required
def feedback(transaction_id):
    label = request.form.get("feedback_label")
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE transactions SET feedback_label = ? WHERE id = ?
        """, (label, transaction_id))
        conn.commit()

    flash("Feedback saved", "success")
    return redirect(url_for("history"))


@app.route("/suspicious")
@login_required
def suspicious():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM transactions
            ORDER BY fraud_probability DESC, id DESC
            LIMIT 50
        """).fetchall()

    return render_template("suspicious.html", rows=rows)


@app.route("/notifications")
@login_required
def notifications():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT 100").fetchall()

    return render_template("notifications.html", rows=rows)


@app.route("/notifications/read/<int:notification_id>")
@login_required
def read_notification(notification_id):
    with get_db_connection() as conn:
        conn.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notification_id,))
        conn.commit()

    return redirect(url_for("notifications"))


@app.route("/risk_profiles")
@login_required
def risk_profiles():
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT
                customer_ref,
                COUNT(*) AS total_transactions,
                ROUND(AVG(fraud_probability), 2) AS avg_probability,
                SUM(CASE WHEN final_prediction = 1 THEN 1 ELSE 0 END) AS fraud_count,
                SUM(CASE WHEN action_recommendation = 'Block' THEN 1 ELSE 0 END) AS blocked_count,
                SUM(CASE WHEN duplicate_flag = 1 THEN 1 ELSE 0 END) AS duplicate_count
            FROM transactions
            WHERE customer_ref IS NOT NULL AND customer_ref <> ''
            GROUP BY customer_ref
            ORDER BY avg_probability DESC, fraud_count DESC
            LIMIT 100
        """).fetchall()

    return render_template("risk_profiles.html", rows=rows)


@app.route("/export_history")
@login_required
def export_history():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM transactions ORDER BY id DESC").fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    csv_data = df.to_csv(index=False)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=transaction_history.csv"}
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
