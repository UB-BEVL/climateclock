
path = r"c:\Users\Shyamli\Documents\IMP\Certificate\BEVL\app.py"
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

search_terms = ["heatmap", "subplots", "st.pyplot", "Solar Radiation"]
for i, line in enumerate(lines):
    for term in search_terms:
        if term in line:
            print(f"{i+1}: {line.strip()[:100]}...")
