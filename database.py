import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'churnlens.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            contract TEXT,
            internet_service TEXT,
            online_security TEXT,
            tech_support TEXT,
            payment_method TEXT,
            paperless_billing TEXT,
            partner TEXT,
            senior_citizen INTEGER,
            churn_probability REAL,
            risk_category TEXT,
            recommendation TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(input_dict, prob, risk, rec):
    init_db()
    conn = get_connection()
    c = conn.cursor()

    contract_map = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
    internet_map = {0: 'No', 1: 'DSL', 2: 'Fiber optic'}
    payment_map = {0: 'Electronic check', 1: 'Mailed check', 2: 'Bank transfer (auto)', 3: 'Credit card (auto)'}

    c.execute('''
        INSERT INTO predictions (
            timestamp, tenure, monthly_charges, total_charges,
            contract, internet_service, online_security, tech_support,
            payment_method, paperless_billing, partner, senior_citizen,
            churn_probability, risk_category, recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        input_dict.get('tenure'),
        input_dict.get('MonthlyCharges'),
        input_dict.get('TotalCharges'),
        contract_map.get(input_dict.get('Contract'), ''),
        internet_map.get(input_dict.get('InternetService'), ''),
        'Yes' if input_dict.get('OnlineSecurity') == 1 else 'No',
        'Yes' if input_dict.get('TechSupport') == 1 else 'No',
        payment_map.get(input_dict.get('PaymentMethod'), ''),
        'Yes' if input_dict.get('PaperlessBilling') == 1 else 'No',
        'Yes' if input_dict.get('Partner') == 1 else 'No',
        input_dict.get('SeniorCitizen'),
        round(prob * 100, 1),
        risk,
        rec
    ))
    conn.commit()
    conn.close()

def get_all_predictions():
    init_db()
    conn = get_connection()
    df = pd.read_sql_query(
        'SELECT * FROM predictions ORDER BY id DESC',
        conn
    )
    conn.close()
    return df

def get_kpis():
    init_db()
    conn = get_connection()
    c = conn.cursor()

    c.execute('SELECT COUNT(*) FROM predictions')
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions WHERE risk_category = 'High'")
    high_risk = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions WHERE risk_category = 'Medium'")
    medium_risk = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions WHERE risk_category = 'Low'")
    low_risk = c.fetchone()[0]

    c.execute('SELECT SUM(monthly_charges) FROM predictions WHERE risk_category = "High"')
    rev_at_risk = c.fetchone()[0] or 0

    c.execute('SELECT AVG(churn_probability) FROM predictions')
    avg_prob = c.fetchone()[0] or 0

    conn.close()
    return {
        'total': total,
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'revenue_at_risk': rev_at_risk,
        'avg_probability': round(avg_prob, 1)
    }

def clear_history():
    init_db()
    conn = get_connection()
    c = conn.cursor()
    c.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
