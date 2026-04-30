from __future__ import annotations
import streamlit as st
import importlib

st.set_page_config(page_title="ClimateClock Launcher", layout="wide")
st.title("ClimateClock — Launcher & Diagnostics")
st.write("Use this lightweight launcher to diagnose startup issues without importing the full application.")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Load full app"):
        with st.spinner("Importing full app (may take a while)..."):
            try:
                full = importlib.import_module("app")
                importlib.reload(full)
                full.main()
            except Exception as e:
                st.error(f"Failed to load full app: {e}")
                import traceback
                st.text(traceback.format_exc())

with col2:
    if st.button("Run import diagnostics"):
        st.info("Attempting to import key dependencies (this runs in the app process).")
        modules = [
            'streamlit','pandas','numpy','matplotlib','fpdf','pythermalcomfort',
            'pvlib','plotly','pydeck','scipy','kaleido'
        ]
        results = {}
        for m in modules:
            try:
                __import__(m)
                results[m] = (True, "OK")
            except Exception as e:
                results[m] = (False, f"{type(e).__name__}: {e}")
        ok = [k for k,v in results.items() if v[0]]
        fail = {k:v[1] for k,v in results.items() if not v[0]}
        st.write(f"Imported OK: {len(ok)}/{len(modules)}")
        if ok:
            st.write(ok)
        if fail:
            st.error("Some imports failed:")
            for k, msg in fail.items():
                st.write(f"- {k}: {msg}")

st.markdown("---")
st.caption("If the launcher loads but the full app still hangs when importing, click 'Run import diagnostics' and paste the results here.")
