import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64, time, os
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ---------------------------
# Config & CSS
# ---------------------------
st.set_page_config(page_title="SpendWise Assistant", layout="wide")

# Set your mascot image path here 👇
assistant_avatar = r"C:\Users\VARUN\OneDrive\Desktop\spendwise_project\mascot.jpg"

# CSS for chat, dark theme, and animation
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.assistant-bubble {
    background:#E8F0FE;
    color:black;
    padding:10px;
    border-radius:10px;
    margin:6px 0;
    display:inline-block;
    max-width:80%;
}
.user-bubble {
    background:#DCF8C6;
    color:black;
    padding:10px;
    border-radius:10px;
    margin:6px 0;
    display:inline-block;
    float:right;
    max-width:80%;
}
.typing {
    display: inline-block;
    position: relative;
    width: 60px;
    height: 20px;
}
.typing div {
    position: absolute;
    top: 0;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #4285F4;
    animation-timing-function: cubic-bezier(0, 1, 1, 0);
}
.typing div:nth-child(1) { left: 6px; animation: typing 0.6s infinite; }
.typing div:nth-child(2) { left: 24px; animation: typing 0.6s infinite 0.2s; }
.typing div:nth-child(3) { left: 42px; animation: typing 0.6s infinite 0.4s; }
@keyframes typing {
 0% { transform: translateY(0); }
 50% { transform: translateY(-6px); }
 100% { transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper Functions
# ---------------------------
def get_avatar_html(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            return f"<img src='data:image/png;base64,{encoded}' width='40' style='border-radius:50%;margin-right:8px;'>"
        except:
            pass
    return "<div style='width:40px;height:40px;border-radius:50%;background:#4285F4;margin-right:8px;text-align:center;line-height:40px;color:white;font-weight:bold;'>S</div>"

def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("Personal_Finance_Dataset.csv")
    return df

def preprocess(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Type'] = df['Type'].astype(str).str.strip().str.capitalize()
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    if 'Category' not in df.columns:
        df['Category'] = 'Other'
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df['Category_encoded'] = df['Category'].astype('category').cat.codes
    df['Type_encoded'] = df['Type'].astype('category').cat.codes
    return df

def monthly_summary(df):
    ms = df.groupby(['YearMonth','Type'])['Amount'].sum().unstack().fillna(0)
    ms['Savings'] = ms['Income'] - ms['Expense']
    ms['Savings_Ratio'] = ms.apply(
        lambda r: (r['Savings'] / r['Income']) if r['Income'] else np.nan, axis=1
    )
    return ms

def forecast_simple(ms, window=3):
    if len(ms) == 0:
        return 0,0,0
    pred_income = ms['Income'].tail(window).mean()
    pred_expense = ms['Expense'].tail(window).mean()
    pred_saving = pred_income - pred_expense
    return float(pred_income), float(pred_expense), float(pred_saving)

def compute_category_allocation(df, base_expense_amount, fixed_rent=None):
    expense_df = df[df['Type']=="Expense"]
    if len(expense_df) == 0:
        return pd.Series(dtype=float)
    dist = expense_df.groupby('Category')['Amount'].sum()
    dist_var = dist[dist.index.str.lower() != 'rent']
    dist_frac = dist_var / dist_var.sum()
    allocation = {cat: round(frac * base_expense_amount,2) for cat, frac in dist_frac.items()}
    if fixed_rent and fixed_rent>0:
        allocation['Rent'] = fixed_rent
    return pd.Series(allocation).sort_values(ascending=False)

def detect_anomalies(df, z_thresh=3.0, iso_cont=0.05, db_eps=0.5):
    X = df[['Amount','Category_encoded','Type_encoded']].astype(float)
    df['Anomaly_ZScore'] = np.abs(stats.zscore(X['Amount'])) > z_thresh
    scaler = StandardScaler()
    iso = IsolationForest(contamination=iso_cont, random_state=42)
    df['Anomaly_IForest'] = iso.fit_predict(scaler.fit_transform(X)) == -1
    db = DBSCAN(eps=db_eps, min_samples=5)
    labels = db.fit_predict(scaler.fit_transform(X))
    df['Anomaly_DBSCAN'] = labels == -1
    df['Consensus_Anomaly'] = df[['Anomaly_ZScore','Anomaly_IForest','Anomaly_DBSCAN']].sum(axis=1) >= 2
    return df

# ---------------------------
# Session Init
# ---------------------------
if 'df' not in st.session_state:
    try:
        st.session_state.df = preprocess(load_dataset(None))
    except:
        st.session_state.df = None
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi Pooh! I'm SpendWise — your budget buddy. Upload your dataset or use default, then tell me your income this month."}
    ]
if 'fixed_rent' not in st.session_state:
    st.session_state.fixed_rent = None

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Data & Controls")
file = st.sidebar.file_uploader("Upload Personal_Finance_Dataset.csv", type=["csv"])
if file:
    try:
        st.session_state.df = preprocess(load_dataset(file))
        st.sidebar.success("✅ Dataset loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.sidebar.write("---")
st.sidebar.write("🏠 Set Fixed Rent")
rent = st.sidebar.number_input("Enter Rent (₹)", min_value=0, value=int(st.session_state.fixed_rent or 0))
if st.sidebar.button("Save Rent"):
    st.session_state.fixed_rent = rent
    st.sidebar.success(f"Fixed rent set to ₹{rent}")

st.sidebar.write("---")
z_thresh = st.sidebar.slider("Z-score threshold", 2.5, 5.0, 3.0)
iso_cont = st.sidebar.slider("IsolationForest contamination", 0.01, 0.2, 0.05)
db_eps = st.sidebar.slider("DBSCAN eps", 0.1, 3.0, 0.5)

# ---------------------------
# Main Chat UI
# ---------------------------
st.title("💸 SpendWise — Interactive Budget Assistant")
col1, col2 = st.columns([2,1])
avatar_html = get_avatar_html(assistant_avatar)

with col1:
    for msg in st.session_state.messages:
        if msg['role'] == 'assistant':
            st.markdown(f"<div style='display:flex;align-items:center'>{avatar_html}<div class='assistant-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.text_input("Type your message (e.g., 'My income is 20000')", key="user_text")
    if st.button("Send"):
        if user_input and st.session_state.df is not None:
            st.session_state.messages.append({"role":"user","content":user_input})
            found_number = None
            for word in user_input.split():
                word_clean = ''.join(ch for ch in word if (ch.isdigit() or ch=='.'))
                if word_clean.isnumeric() or '.' in word_clean:
                    found_number = float(word_clean)
                    break
            ms = monthly_summary(st.session_state.df)
            if found_number:
                income = found_number
                pred_income, pred_exp, _ = forecast_simple(ms)
                base_exp = max(pred_exp - (st.session_state.fixed_rent or 0), 0)
                allocation = compute_category_allocation(st.session_state.df, base_exp, st.session_state.fixed_rent)
                total_exp = allocation.sum()
                savings = max(income - total_exp, 0)
                reply = f"Got it Pooh 🐻! Income: ₹{income:.2f}<br><br>💰 Save: ₹{savings:.2f}<br>🛒 Spend: ₹{total_exp:.2f}<br><br>📊 Split:<br>"
                for cat, amt in allocation.items():
                    reply += f"• {cat}: ₹{amt:.2f}<br>"
                st.session_state.latest_reco = {'income':income,'save':savings,'spend':total_exp,'allocation':allocation}
            else:
                pred_income, _, _ = forecast_simple(ms)
                reply = f"I couldn’t find a number! Based on history, I expect ₹{pred_income:.2f}. Try again!"
            st.session_state.messages.append({"role":"assistant","content":reply})
            st.rerun()

with col2:
    st.subheader("Quick actions")
    if 'latest_reco' in st.session_state and st.session_state.latest_reco:
        lr = st.session_state.latest_reco
        st.metric("Recommended Savings", f"₹{lr['save']:.2f}")
        st.metric("Recommended Expenses", f"₹{lr['spend']:.2f}")
        if st.button("Apply this plan"):
            st.session_state.applied_plan = lr
            st.success("✅ Plan applied successfully! You can now track against it.")
    else:
        st.write("No recommendation yet.")
    st.write("---")
    st.subheader("Simulate GPay Transaction")
    sim_date = st.date_input("Date", value=datetime.today())
    sim_amt = st.number_input("Amount (₹)", 0.0, value=100.0)
    sim_cat = st.text_input("Category", "Food & Drink")
    sim_type = st.selectbox("Type", ["Expense","Income"])
    if st.button("Add simulated transaction"):
        new_row = {'Date': pd.to_datetime(sim_date), 'Transaction Description': f"Simulated {sim_type}",
                   'Category': sim_cat, 'Amount': sim_amt, 'Type': sim_type}
        df2 = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.df = preprocess(df2)
        st.success("Simulated transaction added.")
        st.rerun()

# ---------------------------
# Analytics
# ---------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    ms = monthly_summary(df)
    st.markdown("---")
    st.subheader("Historical Summary")
    st.dataframe(ms.tail(12))

    fig, ax = plt.subplots(figsize=(8,3))
    ms[['Income','Expense','Savings']].plot(ax=ax, marker='o')
    ax.set_title("Income vs Expense vs Savings")
    st.pyplot(fig)

    pred_income, pred_exp, pred_sav = forecast_simple(ms)
    st.subheader("Auto Forecast")
    st.write(f"Pred. Income: ₹{pred_income:.2f}, Pred. Expense: ₹{pred_exp:.2f}, Pred. Saving: ₹{pred_sav:.2f}")

    if 'latest_reco' in st.session_state and st.session_state.latest_reco:
        fig_pie, ax = plt.subplots(figsize=(5,4))
        st.session_state.latest_reco['allocation'].plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig_pie)

    st.subheader("Anomaly Detection")
    df_anom = detect_anomalies(df, z_thresh, iso_cont, db_eps)
    anomalies = df_anom[df_anom['Consensus_Anomaly']]
    st.write(f"Total anomalies found: {len(anomalies)}")
    if len(anomalies)>0:
        st.dataframe(anomalies[['Date','Transaction Description','Category','Amount']].head(10))
        
        # ---------------------------
        # Anomaly Graph
        # ---------------------------
        st.subheader("Anomaly Graph")
        fig_anom, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_anom['Date'], df_anom['Amount'], marker='o', linestyle='-', color='blue', label='Normal')
        ax.scatter(anomalies['Date'], anomalies['Amount'], color='red', label='Anomaly', s=80, zorder=5)
        ax.set_title("Transaction Amounts with Anomalies Highlighted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Amount (₹)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_anom)
