import os
import pandas as pd
import streamlit as st

# Ye file project ke 'pages' folder ke andar rakhni hai (pages/dashboard.py)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(HERE, "scans.csv")

st.set_page_config(page_title="Dashboard", page_icon="📊")
st.title("📊 Scan Dashboard")

if not os.path.exists(LOG_PATH):
    st.info("Abhi koi scan nahi hua. Pehle main page par kuch scan karo.")
    st.stop()

df = pd.read_csv(LOG_PATH)

c1, c2, c3 = st.columns(3)
c1.metric("Total scans", len(df))
c2.metric("Avg confidence", f"{df['confidence'].mean():.0%}")
c3.metric("Agent ne sawaal poocha", f"{df['agent_asked'].mean():.0%}")

st.subheader("Bin ke hisaab se scans")
st.bar_chart(df["bin"].value_counts())

st.subheader("Waste type ke hisaab se")
st.bar_chart(df["final_label"].value_counts())

st.subheader("Recent scans")
st.dataframe(df.tail(20).iloc[::-1], width="stretch")
