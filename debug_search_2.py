
path = r"c:\Users\Shyamli\Documents\IMP\Certificate\BEVL\app.py"
try:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("MATCHES:")
    for i, line in enumerate(lines):
        if "Annual Diurnal" in line or "st.pyplot" in line:
            print(f"L{i+1}: {line.strip()}")
            if len(line) > 500: break # Safety
except Exception as e:
    print(e)
