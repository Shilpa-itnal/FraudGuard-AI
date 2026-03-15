import sqlite3
from werkzeug.security import generate_password_hash

DB_NAME = "fraud_system.db"


def insert_setting(cur, key, value):
    cur.execute("SELECT id FROM settings WHERE setting_key = ?", (key,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO settings (setting_key, setting_value) VALUES (?, ?)",
            (key, value)
        )


def main():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_ref TEXT,
        purchase_value REAL,
        age INTEGER,
        signup_time TEXT,
        purchase_time TEXT,
        account_age_hours REAL,
        purchase_hour INTEGER,
        ml_probability REAL,
        rule_score REAL,
        fraud_probability REAL,
        confidence_band TEXT,
        duplicate_flag INTEGER,
        risk_level TEXT,
        action_recommendation TEXT,
        otp_status TEXT,
        final_prediction INTEGER,
        suspicious_reasons TEXT,
        source_type TEXT,
        created_by TEXT,
        feedback_label TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id INTEGER,
        status TEXT,
        assigned_to TEXT,
        notes TEXT,
        created_at TEXT,
        FOREIGN KEY(transaction_id) REFERENCES transactions(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS case_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id INTEGER,
        note_text TEXT,
        added_by TEXT,
        created_at TEXT,
        FOREIGN KEY(case_id) REFERENCES cases(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        version_tag TEXT,
        trained_at TEXT,
        dataset_name TEXT,
        notes TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setting_key TEXT UNIQUE,
        setting_value TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS list_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        list_type TEXT,
        entry_value TEXT,
        notes TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS otp_verifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id INTEGER,
        otp_code TEXT,
        status TEXT,
        generated_at TEXT,
        verified_at TEXT,
        FOREIGN KEY(transaction_id) REFERENCES transactions(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        message TEXT,
        severity TEXT,
        is_read INTEGER DEFAULT 0,
        created_at TEXT
    )
    """)

    users = [
        ("admin", generate_password_hash("admin123"), "admin"),
        ("analyst", generate_password_hash("analyst123"), "analyst"),
        ("analyst2", generate_password_hash("analyst2123"), "analyst")
    ]

    for username, password_hash, role in users:
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash, role)
            )

    insert_setting(cur, "review_threshold", "0.40")
    insert_setting(cur, "otp_threshold", "0.65")
    insert_setting(cur, "block_threshold", "0.85")
    insert_setting(cur, "high_purchase_multiplier", "2.0")
    insert_setting(cur, "new_account_hours", "24")
    insert_setting(cur, "duplicate_hours_window", "2")

    cur.execute("SELECT id FROM model_versions WHERE version_tag = ?", ("v2",))
    if not cur.fetchone():
        cur.execute("""
            INSERT INTO model_versions (model_name, version_tag, trained_at, dataset_name, notes)
            VALUES (?, ?, datetime('now'), ?, ?)
        """, ("RandomForest_v2", "v2", "ecommerce_fraud.csv", "Advanced upgraded version"))

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


if __name__ == "__main__":
    main()
