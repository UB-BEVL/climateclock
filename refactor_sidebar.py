import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Header Block
old_header = r"\"\"\"\n            <div style='text-align: center; margin-bottom: 1\.5rem; padding: 0\.9rem 0\.75rem;'>\n                <div style='font-size: 2\.1rem; margin-bottom: 0\.35rem;'>&#127780;</div>\n                <p style='font-size: 1rem; font-weight: 700; color:#e2e8f0; margin-bottom: 0\.15rem; letter-spacing: -0\.01em;'>\{loc_label\}</p>\n                <p style='color: #c5cbd8; font-size: 0\.9rem; font-weight: 500;'>Quick access tools</p>\n            </div>\n            \"\"\""
new_header = r"\"\"\"\n            <div style='display: flex; align-items: center; margin-bottom: 0.5rem; padding: 0.2rem;'>\n                <div style='font-size: 1.8rem; margin-right: 0.5rem;'>&#127780;</div>\n                <div>\n                    <h3 style='margin:0; font-size:1.0rem; padding:0; color:#e2e8f0; font-weight: 600;'>Climate Analysis Pro</h3>\n                    <p style='margin:0; font-size:0.8rem; color:#94a3b8; font-weight:500;'>{loc_label}</p>\n                </div>\n            </div>\n            \"\"\""
text = re.sub(old_header, new_header, text)

# Remove dividers before navigation
text = text.replace("st.divider()\n\n        epw_loaded", "epw_loaded")
text = text.replace("st.markdown(\"### Visualize weather file\")\n        ", "")

# We have st.divider() at 2272. Replace Export Report section
old_export_start = 'st.divider()\n        st.markdown("**?? Export Report**")'
new_export_start = 'st.markdown("<br>", unsafe_allow_html=True)\n        with st.expander("?? Export PDF Report", expanded=False):'
text = text.replace(old_export_start, new_export_start)

# Since we wrapped inside an expander, we need to indent everything that was under it.
# To do this safely, I will just do a string replacement for the Troubleshooting part too

old_trouble = 'st.divider()\n        st.markdown("### ?? Troubleshooting")\n\n        if st.button("Reset Session & Try Again"):'
new_trouble = 'st.markdown("<br>", unsafe_allow_html=True)\n        if st.button("?? Reset Session", use_container_width=True):'
text = text.replace(old_trouble, new_trouble)

# Remove st.markdown("### Filters and units")
text = text.replace('st.markdown("### Filters and units")\n    with st.expander("Filters and units", expanded=False):', 'with st.expander("?? Settings & filters", expanded=False):')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(text)
