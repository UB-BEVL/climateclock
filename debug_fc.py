
path = r"c:\Users\Shyamli\Documents\IMP\Certificate\BEVL\app.py"
try:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    found = False
    for i, line in enumerate(lines):
        if "fc" in line and not "def" in line and not "if" in line and "=" in line: # simplistic check to find usages or assignments
             pass
    
    # Just print lines with "fc." usage
    for i, line in enumerate(lines):
        if "fc." in line:
            print(f"Line {i+1}: {line.rstrip()}")
            found = True
            
    if not found:
        print("No matches for 'fc.' found.")
            
except Exception as e:
    print(f"Error: {e}")
