
path = r"c:\Users\Shyamli\Documents\IMP\Certificate\BEVL\app.py"
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def render_" in line and "sensor" in line.lower():
        print(f"{i+1}: {line.strip()}")
    if "Sensor Comparison" in line:
        print(f"Text match at {i+1}: {line.strip()}")
