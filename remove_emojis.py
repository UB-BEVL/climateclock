with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

changes = {
    '??? Select weather file': 'Select weather file',
    '?? Dashboard': 'Dashboard',
    '?? Live Data vs EPW': 'Live Data vs EPW',
    '?? Sensor Comparison': 'Sensor Comparison',
    '?? Short-Term Prediction (24–72h)': 'Short-Term Prediction (24–72h)',
    '?? Short-Term Prediction (24–72h)': 'Short-Term Prediction (24–72h)',
    '?? Future Climate (2050 / 2080 SSP)': 'Future Climate (2050 / 2080 SSP)',
    '?? Short-term prediction & future climate are coming soon': 'Short-term prediction & future climate are coming soon'
}

for old, new in changes.items():
    text = text.replace(old, new)
    
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(text)
