import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Frontline Safety AI", layout="wide")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"


# -----------------------------
# LOAD DATA (cached = fast)
# -----------------------------
@st.cache_data
def load_data():
    deptA = pd.read_excel("DeptA_incidents_test_dataset_100_rows.xlsx")
    deptB = pd.read_excel("DeptB_incidents_test_dataset_100_rows.xlsx")
    deptC = pd.read_excel("DeptC_incidents_test_dataset_100_rows.xlsx")
    deptD = pd.read_csv("DeptD_incidents_test_dataset_100_rows.csv")
    return [deptA, deptB, deptC, deptD]


# -----------------------------
# FIND COLUMN (flexible)
# -----------------------------
def find_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key in col.lower():
                return col
    return None


# -----------------------------
# GET ADDRESSES
# -----------------------------
@st.cache_data
def get_addresses():
    dfs = load_data()
    all_addresses = []

    for df in dfs:
        col = find_column(df, ["address", "location", "site"])
        if col:
            all_addresses.append(df[col])

    if not all_addresses:
        return ["No addresses found"]

    addresses = pd.concat(all_addresses).dropna().astype(str).unique().tolist()
    return sorted(addresses)[:100]


# -----------------------------
# EXTRACT INCIDENTS
# -----------------------------
def extract_incidents(df, address):

    address_col = find_column(df, ["address", "location", "site"])
    date_col = find_column(df, ["date"])
    type_col = find_column(df, ["incident", "type"])
    actor_col = find_column(df, ["actor", "person"])
    desc_col = find_column(df, ["description", "details", "notes"])

    if not address_col:
        return []

    matches = df[df[address_col].astype(str).str.contains(address, case=False, na=False)]

    incidents = []

    for _, row in matches.head(5).iterrows():

        incident = f"""Date: {row.get(date_col, "")}
Address: {row.get(address_col, "")}
Incident: {row.get(type_col, "")}
Actor: {row.get(actor_col, "")}
Details: {row.get(desc_col, "")}"""

        incidents.append(incident)

    return incidents


# -----------------------------
# UI
# -----------------------------
st.title("🛡️ Frontline Safety AI – Address Risk Assistant")

st.markdown("Search multiple departmental datasets and generate a safety briefing.")

addresses = get_addresses()

selected_address = st.selectbox("Select Address", addresses)


# -----------------------------
# SEARCH BUTTON
# -----------------------------
if st.button("🔍 Search Incidents"):

    dfs = load_data()

    all_incidents = []
    for df in dfs:
        all_incidents += extract_incidents(df, selected_address)

    if not all_incidents:
        st.warning("No incidents found.")
    else:
        st.session_state["incidents"] = all_incidents

        st.subheader("Incident Data")
        st.text("\n\n".join(all_incidents))

        # Risk scoring
        count = len(all_incidents)
        if count >= 6:
            risk = "HIGH"
        elif count >= 3:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        st.session_state["risk"] = risk


# -----------------------------
# AI BUTTON
# -----------------------------
if st.button("🤖 Generate AI Briefing"):

    incidents = st.session_state.get("incidents", [])
    risk = st.session_state.get("risk", "LOW")

    if not incidents:
        st.warning("Run search first.")
    else:

        incident_text = "\n\n".join(incidents)

        prompt = f"""
Generate EXACTLY this format:

RISK LEVEL: {risk}

Key Risks
Summarise key risks.

Recommendation
Provide safety advice.

Incidents:
{incident_text}
"""

        try:
            response = requests.post(API_URL, json={"inputs": prompt}, timeout=10)

            if response.status_code == 200:
                ai_text = response.json()[0]["generated_text"]
            else:
                ai_text = f"""RISK LEVEL: {risk}

Key Risks
Multiple incidents recorded.

Recommendation
Proceed with caution."""

        except:
            ai_text = f"""RISK LEVEL: {risk}

Key Risks
Multiple incidents recorded.

Recommendation
Proceed with caution."""

        st.subheader("AI Risk Briefing")
        st.text(f"{ai_text}\n\n{incident_text}")
