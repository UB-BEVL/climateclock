import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

pattern = r'st\.session_state\.setdefault\("pdf_figures", \{\}\)\[(.+?)\]\s*=\s*(.+?)\s*$'
regex = re.compile(pattern, re.MULTILINE)

new_text = regex.sub(r'_add_manual_pdf_figure(\1, \2)', text)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print(f'Replaced {len(regex.findall(text))} occurrences')
