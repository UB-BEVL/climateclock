
path = r"c:\Users\Shyamli\Documents\IMP\Certificate\BEVL\app.py"
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def render_trends_page" in line:
        print(f"{i+1}: {line.strip()}")
