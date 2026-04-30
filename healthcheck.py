import streamlit as st
import platform

st.set_page_config(page_title="Healthcheck", layout="centered")
st.title("Healthcheck — ClimateClock")
st.write("If you see this, the Streamlit process started and served a minimal app.")
st.write("Python:", platform.python_version())
st.write("Platform:", platform.platform())
st.write("Environment variables (partial):")
for k in sorted([k for k in list(dict(**__import__('os').environ).keys()) if k.lower().startswith(('streamlit','git','user','path'))][:20]):
    st.write(f"- {k}: {__import__('os').environ.get(k)}")
