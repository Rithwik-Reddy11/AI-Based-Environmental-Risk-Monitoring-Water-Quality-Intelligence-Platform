import os, sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_pipeline import load_data
from src.model import build_model, FEATURES
from src.risk_engine import calculate_wqi, risk_from_probability, risk_score, what_if_message
from src.explainability import explain

st.set_page_config(page_title="Environmental Risk Intelligence", page_icon="🌊", layout="wide")

@st.cache_data

def get_data():
    df = load_data()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["WQI"] = df.apply(lambda r: calculate_wqi(r.pH,r.Dissolved_Oxygen,r.Nitrate,r.Turbidity,r.Water_Temperature), axis=1)
    df["Risk_Score"] = 100 - df["WQI"]
    return df

@st.cache_resource

def get_model(df):
    model = build_model(); model.fit(df[FEATURES], df["Pollution_Flag"].astype(int)); return model

df = get_data(); model = get_model(df)

st.title("🌊 AI-Based Environmental Risk Monitoring & Water Quality Intelligence")
st.caption("A decision-support demo for water-quality monitoring. The included dataset is synthetic; predictions are not regulatory measurements.")

# Sidebar
st.sidebar.header("Dashboard Controls")
stations = sorted(df["Station"].unique())
station = st.sidebar.selectbox("Monitoring station", ["All stations"] + stations)
if station != "All stations": view = df[df.Station == station].copy()
else: view = df.copy()

# KPIs
latest = view.sort_values("Timestamp").iloc[-1]
risk_prob = float(model.predict_proba(latest[FEATURES].to_frame().T)[0,1]*100)
label, level = risk_from_probability(risk_prob)
score = risk_score(risk_prob, latest.WQI)
high_count = int((view["Risk_Score"] >= 60).sum())

c1,c2,c3,c4 = st.columns(4)
c1.metric("Water Quality Index", f"{latest.WQI:.1f}/100")
c2.metric("AI Pollution Probability", f"{risk_prob:.1f}%")
c3.metric("Environmental Risk Score", f"{score:.1f}/100")
c4.metric("High-Risk Observations", high_count)

st.divider()

p1,p2 = st.columns(2)
with p1:
    st.subheader("📈 Historical Water Quality")
    trend = view.set_index("Timestamp").resample("D")["WQI"].mean().reset_index()
    st.plotly_chart(px.line(trend,x="Timestamp",y="WQI",title="Daily average WQI"), use_container_width=True)
with p2:
    st.subheader("🧪 Water Parameters")
    param = st.selectbox("Parameter", ["pH","Nitrate","Turbidity","Dissolved_Oxygen","Water_Temperature","Conductivity"])
    trend2 = view.set_index("Timestamp").resample("D")[param].mean().reset_index()
    st.plotly_chart(px.line(trend2,x="Timestamp",y=param,title=f"Daily {param}"), use_container_width=True)

st.divider()

left,right = st.columns([1,1])
with left:
    st.subheader("🔮 Site Risk Prediction")
    industry = st.selectbox("Industry type", sorted(df.Industry_Type.unique()))
    ph = st.slider("pH",3.0,10.0,7.0,.1)
    nitrate = st.number_input("Nitrate (mg/L)",0.0,100.0,5.0,.1)
    temp = st.slider("Water temperature (°C)",5.0,40.0,22.0,.5)
    turb = st.slider("Turbidity (NTU)",0.0,150.0,10.0,1.0)
    do = st.slider("Dissolved oxygen (mg/L)",0.0,14.0,7.0,.1)
    cond = st.slider("Conductivity (µS/cm)",50.0,2000.0,300.0,10.0)
    Xnew = pd.DataFrame([[industry,ph,nitrate,temp,turb,do,cond]], columns=FEATURES)
    if st.button("🚨 Analyze Environmental Risk", type="primary"):
        prob = float(model.predict_proba(Xnew)[0,1]*100)
        lab, lvl = risk_from_probability(prob); wqi=calculate_wqi(ph,do,nitrate,turb,temp); rs=risk_score(prob,wqi)
        st.metric("Pollution probability",f"{prob:.1f}%")
        st.metric("WQI",f"{wqi:.1f}/100")
        st.metric("Risk score",f"{rs:.1f}/100")
        st.warning(f"{lab} — {lvl}") if lvl in ["HIGH","CRITICAL"] else st.info(f"{lab} — {lvl}")

with right:
    st.subheader("🧠 Explainability")
    sample = view[FEATURES + ["Pollution_Flag"]].dropna().sample(min(250,len(view)),random_state=42)
    importance = explain(model, sample[FEATURES], sample["Pollution_Flag"])
    st.dataframe(importance.head(10), use_container_width=True, hide_index=True)
    st.caption("Feature importance is model-agnostic permutation importance. SHAP can be added when deploying with the optional shap dependency.")

st.divider()

st.subheader("🗺️ Geographic Risk Monitoring")
# Deterministic demo coordinates; clearly marked as simulated monitoring locations.
coords = {s:(12.9716+(i%5)*.035,77.5946+(i//5)*.045) for i,s in enumerate(stations)}
map_df = df.groupby("Station").tail(1).copy(); map_df["Latitude"] = map_df.Station.map(lambda s:coords[s][0]); map_df["Longitude"] = map_df.Station.map(lambda s:coords[s][1]); map_df["AI_Risk"] = map_df[FEATURES].apply(lambda r:model.predict_proba(pd.DataFrame([r],columns=FEATURES))[0,1]*100,axis=1)
try:
    import folium
    from streamlit_folium import st_folium
    m=folium.Map(location=[12.99,77.62],zoom_start=11)
    for _,r in map_df.iterrows():
        color="red" if r.AI_Risk>=60 else "orange" if r.AI_Risk>=40 else "green"
        folium.CircleMarker([r.Latitude,r.Longitude],radius=8,color=color,fill=True,fill_opacity=.8,popup=f"{r.Station}: {r.AI_Risk:.1f}% risk").add_to(m)
    st_folium(m,width=None,height=450)
    st.caption("Map coordinates are simulated for demonstration and are not real sampling locations.")
except ImportError:
    st.info("Install folium and streamlit-folium to enable the interactive map.")

st.divider()

st.subheader("🚨 Risk Alerts")
alerts = map_df[map_df.AI_Risk>=60][["Station","AI_Risk","WQI","Dissolved_Oxygen","Turbidity","Nitrate"]].sort_values("AI_Risk",ascending=False)
if len(alerts):
    for _,r in alerts.iterrows(): st.error(f"🔴 {r.Station}: {r.AI_Risk:.1f}% pollution probability | WQI {r.WQI:.1f} | DO {r.Dissolved_Oxygen:.1f} | Turbidity {r.Turbidity:.1f}")
else: st.success("No high-risk stations in the current simulated snapshot.")

st.divider()

st.subheader("🧪 What-If Environmental Simulator")
base_row = latest.copy()
changed_do = st.slider("Simulate dissolved oxygen",0.0,14.0,float(base_row.Dissolved_Oxygen),.1)
changed_turb = st.slider("Simulate turbidity",0.0,150.0,float(base_row.Turbidity),1.0)
base_prob = float(model.predict_proba(base_row[FEATURES].to_frame().T)[0,1]*100)
sim = base_row[FEATURES].copy(); sim["Dissolved_Oxygen"]=changed_do; sim["Turbidity"]=changed_turb
sim_prob=float(model.predict_proba(sim.to_frame().T)[0,1]*100)
st.metric("Scenario pollution probability",f"{sim_prob:.1f}%",f"{sim_prob-base_prob:+.1f} pts")
st.write(what_if_message(base_prob,sim_prob))

st.caption("⚠️ This application is an educational/portfolio prototype. Validate against certified field measurements before any environmental or regulatory decision.")
