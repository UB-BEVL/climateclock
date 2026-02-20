
with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

output = []
for i, line in enumerate(lines):
    if "st.plotly_chart" in line:
        output.append(f"{i+1}: {line.strip()}")

with open("plotly_charts.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))
