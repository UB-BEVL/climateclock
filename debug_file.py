
path = r"c:\Users\Shyamli\Documents\IMP\Certificate\BEVL\app.py"
try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    if len(lines) >= 5208:
        print(f"Line 5208: {lines[5207]}")
    else:
        print("File too short for line 5208")
    
    # Check for header definition in render_solar_page
    # Identify start of function
    start_line = -1
    for i, line in enumerate(lines):
        if "def render_solar_page():" in line:
            start_line = i
            print(f"render_solar_page starts at line {i+1}")
            break
            
    if start_line != -1:
        # Check first 20 lines of function
        for i in range(start_line, min(start_line + 20, len(lines))):
            print(f"Line {i+1}: {lines[i].rstrip()}")
            
except Exception as e:
    print(f"Error: {e}")
