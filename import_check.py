import sys, time
modules = ['streamlit','pandas','numpy','matplotlib','fpdf','pythermalcomfort','pvlib','plotly']
print(f"Checking {len(modules)} modules...", flush=True)
for m in modules:
    try:
        __import__(m)
        print(f"OK: {m}", flush=True)
    except Exception as e:
        print(f"FAIL: {m} -> {type(e).__name__}: {e}", flush=True)
    time.sleep(0.05)
