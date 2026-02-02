
path = r"c:\Users\Shyamli\Documents\IMP\Certificate\BEVL\app.py"
try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    found = False
    for i, line in enumerate(lines):
        if "fepw" in line or "SCENARIO_MAP" in line:
            print(f"Line {i+1}: {line.rstrip()}")
            found = True
    if not found:
        print("No matches for 'fepw' or 'SCENARIO_MAP' found.")
            
except Exception as e:
    print(f"Error: {e}")
