import re
import os

filepath = 'app.py'

# Read the contents of app.py
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace png with svg
text = re.sub(r'format="png"', 'format="svg"', text)
text = re.sub(r'mime="image/png"', 'mime="image/svg+xml"', text)
text = re.sub(r'"image/png"', '"image/svg+xml"', text)
text = re.sub(r'\.png"', '.svg"', text)
text = re.sub(r'\(PNG\)', '(SVG)', text)
text = re.sub(r'as PNG', 'as SVG', text)
text = re.sub(r'png_bytes', 'svg_bytes', text)

# Write out the modified content
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Replaced PNG downloads with SVG in {filepath}.")
