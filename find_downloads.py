
with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

output = []
for i, line in enumerate(lines):
    if "download_button" in line:
        # Get 2 lines of context before
        start = max(0, i-2)
        context = lines[start:i+1]
        for j, cl in enumerate(context):
            output.append(f"{start+j+1}: {cl.strip()}")
        output.append("-" * 20)

with open("downloads_lines.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))
