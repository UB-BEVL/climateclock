CLIMATE_INTELLIGENCE_CSS = r'''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800&display=swap');

:root {
    --ci-sidebar-width: 248px;
    --ci-bg: #faf6f1;
    --ci-bg-elevated: #ffffff;
    --ci-panel: #ffffff;
    --ci-panel-2: #f5ede4;
    --ci-border: rgba(139, 90, 43, 0.10);
    --ci-border-strong: rgba(192, 130, 90, 0.35);
    --ci-text: #3d2e22;
    --ci-muted: #7a6555;
    --ci-subtle: #a89585;
    --ci-primary: #c0825a;
    --ci-primary-dark: #f8efe6;
    --ci-amber: #c49234;
    --ci-coral: #b8674a;
    --ci-blue: #c0825a;
    --ci-shadow: 0 4px 16px rgba(100, 60, 20, 0.06);
    --ci-font: 'DM Sans', 'Inter', 'Segoe UI', system-ui, sans-serif;
}

body::before,
body::after {
    display: none !important;
}

html,
body,
.main,
.block-container,
.main > .block-container {
    background: var(--ci-bg) !important;
    color: var(--ci-muted) !important;
    font-family: var(--ci-font) !important;
}

[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {
    display: none !important;
}

[data-testid="stElementContainer"]:has(#station-picker) {
    display: none !important;
}

.block-container {
    max-width: 1520px;
    padding: 0.85rem 1.5rem 2.5rem !important;
}

h1, h2, h3, h4 {
    color: var(--ci-text) !important;
    letter-spacing: -0.01em !important;
    font-family: var(--ci-font) !important;
}

h1 {
    font-size: 2rem !important;
    line-height: 1.12 !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 1.45rem !important;
    font-weight: 600 !important;
}

h3 {
    font-size: 1.12rem !important;
    font-weight: 600 !important;
}

p, span, li, td, th, label, div {
    font-family: var(--ci-font) !important;
}

section[data-testid="stSidebar"][aria-expanded="true"],
section[data-testid="stSidebar"][aria-expanded="true"] > div,
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child {
    width: var(--ci-sidebar-width) !important;
    min-width: var(--ci-sidebar-width) !important;
    max-width: var(--ci-sidebar-width) !important;
    background: linear-gradient(180deg, #f5ede4 0%, #efe5d8 100%) !important;
    border-right: 1px solid rgba(139, 90, 43, 0.08) !important;
    box-shadow: 4px 0 20px rgba(100, 60, 20, 0.04) !important;
    font-family: var(--ci-font) !important;
}

section[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
    align-self: flex-start !important;
    overflow: hidden !important;
    z-index: 999 !important;
}

section[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] > div {
    height: 100vh !important;
    overflow: hidden !important;
    scrollbar-width: none;
}

[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    padding: 1rem 0.9rem 1rem !important;
}

[data-testid="stSidebarHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
}

@media (min-width: 769px) {
    [data-testid="stMain"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] > section.main,
    div[data-testid="stAppViewContainer"] section.main {
        margin-left: var(--ci-sidebar-width) !important;
        width: calc(100% - var(--ci-sidebar-width)) !important;
        max-width: calc(100% - var(--ci-sidebar-width)) !important;
    }
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[title="Close sidebar"],
button[title="Open sidebar"],
button[title*="sidebar" i],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label*="sidebar" i] {
    display: none !important;
}

section[data-testid="stSidebar"][aria-expanded="false"],
section[data-testid="stSidebar"][aria-expanded="false"] > div,
section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child,
section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {
    width: var(--ci-sidebar-width) !important;
    min-width: var(--ci-sidebar-width) !important;
    max-width: var(--ci-sidebar-width) !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
    left: 0 !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] *,
section[data-testid="stSidebar"][aria-expanded="false"] *::before,
section[data-testid="stSidebar"][aria-expanded="false"] *::after {
    transform: translateX(0) !important;
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.15rem !important;
}

/* Hide sidebar scrollbar */
[data-testid="stSidebar"] ::-webkit-scrollbar {
    display: none !important;
}

.cc-sidebar-brand-zone {
    display: block;
    margin: 0 0 2.25rem;
}

.cc-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 0 0 0.45rem;
    padding: 0.2rem 0.1rem;
}

.cc-sidebar-mark {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, rgba(192, 130, 90, 0.15), rgba(196, 146, 52, 0.10));
    border: 1px solid rgba(192, 130, 90, 0.25);
    color: var(--ci-text);
    font-weight: 800;
    font-size: 0.62rem;
    flex-shrink: 0;
}

.cc-sidebar-title {
    color: var(--ci-text);
    font-size: 0.85rem;
    font-weight: 700;
    line-height: 1.15;
}

.cc-sidebar-subtitle {
    color: var(--ci-muted);
    font-size: 0.68rem;
    margin-top: 0.08rem;
    max-width: 155px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.cc-sidebar-status {
    display: grid;
    grid-template-columns: 7px 1fr;
    gap: 0.4rem;
    align-items: center;
    padding: 0 0.1rem 0.35rem;
    margin-bottom: 0;
    border-bottom: 1px solid rgba(139, 90, 43, 0.08);
}

.cc-sidebar-status strong {
    display: block;
    color: var(--ci-text);
    font-size: 0.68rem;
    line-height: 1.15;
}

.cc-sidebar-status span:not(.cc-status-dot) {
    display: block;
    color: var(--ci-subtle);
    font-size: 0.6rem;
    line-height: 1.25;
    margin-top: 0.05rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.cc-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: #c4b5a6;
}

.cc-status-dot.is-ready {
    background: #8db87a;
    box-shadow: 0 0 0 2px rgba(141, 184, 122, 0.2);
}

/* Sidebar nav buttons: flat warm style */
[data-testid="stSidebar"] .stButton > button {
    min-height: 33px !important;
    height: auto !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.38rem 0.6rem !important;
    margin-bottom: 1px !important;
    box-shadow: none !important;
    transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease !important;
    font-family: var(--ci-font) !important;
}

/* Inactive nav items */
section[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"],
section[data-testid="stSidebar"] .stButton button[data-testid*="secondary"] {
    color: var(--ci-muted, #7a6555) !important;
    background: transparent !important;
    border: none !important;
    border-color: transparent !important;
    box-shadow: none !important;
}

/* Inactive hover */
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"]:hover,
section[data-testid="stSidebar"] .stButton button[data-testid*="secondary"]:hover {
    color: var(--ci-text, #3d2e22) !important;
    background: rgba(192, 130, 90, 0.06) !important;
    border: none !important;
    transform: none !important;
}

/* Active nav item */
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"],
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] .stButton button[data-testid*="primary"] {
    color: var(--ci-text, #3d2e22) !important;
    background: rgba(192, 130, 90, 0.10) !important;
    border: none !important;
    border-left: 3px solid var(--ci-primary, #c0825a) !important;
    box-shadow: none !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stButton button:disabled,
section[data-testid="stSidebar"] .stButton button[disabled] {
    color: rgba(122, 101, 85, 0.4) !important;
    background: transparent !important;
    border-color: transparent !important;
    cursor: not-allowed !important;
}

.cc-sidebar-hint {
    margin: 0.7rem 0 0.45rem;
    padding: 0.6rem 0.65rem;
    border-radius: 8px;
    background: rgba(192, 130, 90, 0.06);
    border: 1px solid rgba(192, 130, 90, 0.12);
    color: #7a6555;
    font-size: 0.72rem;
    line-height: 1.42;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255, 253, 250, 0.9) !important;
    border: 1px solid rgba(139, 90, 43, 0.08) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

.app-header {
    background: rgba(255, 253, 250, 0.97) !important;
    border-bottom: 1px solid rgba(139, 90, 43, 0.08) !important;
    box-shadow: 0 2px 12px rgba(100, 60, 20, 0.04) !important;
    margin: -0.85rem -1.5rem 0.75rem !important;
    padding: 1rem 1.5rem !important;
}

.header-logo {
    max-height: 58px !important;
}

.header-title {
    font-size: 1.28rem !important;
    color: #3d2e22 !important;
    font-family: var(--ci-font) !important;
}

.header-location {
    font-size: 0.86rem !important;
    color: #7a6555 !important;
}

.header-ub {
    max-height: 50px !important;
}

.cc-station-intro {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.5rem;
    padding: 0.2rem 0 0.65rem;
    border-bottom: 1px solid rgba(139, 90, 43, 0.08);
    margin-bottom: 0.65rem;
}

.cc-station-intro h1 {
    margin: 0 !important;
    font-size: 1.55rem !important;
}

.cc-station-intro p:not(.cc-eyebrow) {
    margin: 0.35rem 0 0;
    max-width: 720px;
    color: var(--ci-muted);
    font-size: 0.92rem;
    line-height: 1.45;
}

.cc-map-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.35rem 0 0.35rem;
}

.cc-map-heading h3 {
    margin: 0 !important;
    font-size: 1.02rem !important;
}

.cc-map-heading p {
    margin: 0.2rem 0 0;
    color: #7a6555;
    font-size: 0.92rem;
    line-height: 1.45;
}

div[data-testid="stFileUploader"] {
    margin-bottom: 0.25rem;
}

.st-key-main_epw_upload_primary {
    margin-bottom: 0 !important;
}

div[data-testid="stFileUploader"] section {
    background: rgba(255, 253, 250, 0.9) !important;
    border: 1px dashed rgba(192, 130, 90, 0.3) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

.cc-source-divider {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    align-items: center;
    gap: 0.85rem;
    margin: 1.05rem 0 1rem;
    color: #7a6555;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.cc-source-divider::before,
.cc-source-divider::after {
    content: "";
    height: 1px;
    background: rgba(139, 90, 43, 0.10);
}

.cc-source-divider span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 2.4rem;
    min-height: 1.45rem;
    border-radius: 999px;
    border: 1px solid rgba(139, 90, 43, 0.10);
    background: rgba(255, 253, 250, 0.96);
}

.cc-hero-panel,
.cc-page-intro,
.cc-panel,
.cc-mini-card,
.cc-export-note {
    background: linear-gradient(180deg, rgba(255, 253, 250, 0.98), rgba(250, 246, 241, 0.98));
    border: 1px solid var(--ci-border);
    border-radius: 10px;
    box-shadow: var(--ci-shadow);
}

.cc-hero-panel {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(240px, 360px);
    gap: 1.5rem;
    align-items: end;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1rem;
}

.cc-page-intro {
    padding: 1.15rem 1.25rem;
    margin-bottom: 1rem;
}

.cc-eyebrow {
    margin: 0 0 0.35rem;
    color: var(--ci-primary);
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.cc-hero-panel h1,
.cc-page-intro h1 {
    margin: 0 !important;
}

.cc-hero-copy,
.cc-page-intro p:not(.cc-eyebrow) {
    max-width: 780px;
    margin: 0.55rem 0 0;
    color: var(--ci-muted);
    font-size: 0.98rem;
    line-height: 1.55;
}

.cc-hero-meta {
    display: grid;
    gap: 0.45rem;
}

.cc-hero-meta span {
    display: block;
    padding: 0.55rem 0.65rem;
    border-radius: 8px;
    background: rgba(192, 130, 90, 0.04);
    border: 1px solid rgba(139, 90, 43, 0.08);
    color: #5a4435;
    font-size: 0.84rem;
}

.cc-panel {
    padding: 1rem;
    margin-bottom: 1rem;
}

.cc-panel-head h3 {
    margin: 0 !important;
    color: var(--ci-text) !important;
    font-size: 1.05rem !important;
}

.cc-panel-head p {
    margin: 0.35rem 0 0;
    color: var(--ci-muted);
    font-size: 0.88rem;
    line-height: 1.45;
}

.cc-summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}

.cc-summary-grid div,
.cc-mini-card,
.cc-export-note {
    padding: 0.78rem;
}

.cc-summary-grid span,
.cc-mini-card span,
.cc-export-note span {
    display: block;
    color: var(--ci-subtle);
    font-size: 0.78rem;
    line-height: 1.4;
}

.cc-summary-grid strong,
.cc-mini-card strong,
.cc-export-note strong {
    display: block;
    color: var(--ci-text);
    font-size: 0.95rem;
    margin-top: 0.2rem;
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255, 253, 250, 0.98), rgba(250, 246, 241, 0.98));
    border: 1px solid var(--ci-border);
    border-radius: 10px;
    padding: 0.9rem 0.95rem;
    box-shadow: 0 2px 10px rgba(100, 60, 20, 0.05);
}

[data-testid="stMetricLabel"] {
    color: var(--ci-subtle) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    color: var(--ci-text) !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
}

div[role="radiogroup"][aria-label="Dashboard view"] {
    display: flex !important;
    flex-wrap: wrap;
    gap: 0.5rem !important;
    padding: 0.45rem !important;
    margin-bottom: 1rem;
    background: rgba(255, 253, 250, 0.9);
    border: 1px solid var(--ci-border);
    border-radius: 10px;
}

div[role="radiogroup"][aria-label="Dashboard view"] label {
    min-height: 38px;
    padding: 0.48rem 0.75rem !important;
    border: 1px solid transparent;
    border-radius: 8px;
    color: var(--ci-muted);
    font-weight: 650;
    transition: all 0.2s ease;
}

div[role="radiogroup"][aria-label="Dashboard view"] label > div:first-child {
    display: none !important;
}

div[role="radiogroup"][aria-label="Dashboard view"] label:has(input:checked) {
    color: var(--ci-text) !important;
    background: rgba(192, 130, 90, 0.10) !important;
    border-color: rgba(192, 130, 90, 0.25);
    box-shadow: inset 0 -2px 0 var(--ci-primary);
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 253, 250, 0.9) !important;
    border: 1px solid var(--ci-border) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    padding: 0.45rem !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--ci-muted) !important;
    font-weight: 650 !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(192, 130, 90, 0.10) !important;
    color: var(--ci-text) !important;
    box-shadow: inset 0 -2px 0 var(--ci-primary) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 1rem 0 0 !important;
    overflow: visible !important;
    min-height: 420px;
}

.stTabs [data-baseweb="tab-panel"] > div,
.stTabs > div:nth-of-type(2) > div > div,
[role="tabpanel"] {
    overflow: visible !important;
}

.js-plotly-plot {
    border-radius: 10px !important;
    border: 1px solid rgba(139, 90, 43, 0.08);
    box-shadow: 0 4px 18px rgba(100, 60, 20, 0.05) !important;
    overflow: hidden;
}

[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
    min-height: 420px !important;
}

[data-testid="stPlotlyChart"] svg.main-svg {
    min-height: 420px !important;
}

.stButton button,
.stDownloadButton button {
    border-radius: 8px !important;
    min-height: 38px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    font-family: var(--ci-font) !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton button {
    background: transparent !important;
    color: var(--ci-muted, #7a6555) !important;
    border: 1px solid rgba(139, 90, 43, 0.18) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.75rem !important;
    min-height: 32px !important;
}

.stDownloadButton button:hover {
    background: rgba(192, 130, 90, 0.08) !important;
    border-color: rgba(192, 130, 90, 0.35) !important;
    color: var(--ci-text, #3d2e22) !important;
}

.stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #c0825a, #b5945e) !important;
    color: #ffffff !important;
    border: 1px solid rgba(139, 90, 43, 0.10) !important;
}

.stButton button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #b07548, #a88552) !important;
    box-shadow: 0 4px 12px rgba(192, 130, 90, 0.25) !important;
}

.stButton button[data-testid="baseButton-secondary"] {
    background: rgba(192, 130, 90, 0.05) !important;
    color: var(--ci-text) !important;
    border: 1px solid rgba(139, 90, 43, 0.10) !important;
}

.stButton button[data-testid="baseButton-secondary"]:hover {
    background: rgba(192, 130, 90, 0.10) !important;
}

.cc-export-note {
    margin-top: 1rem;
}

.cc-pdf-capture-screen {
    background: linear-gradient(180deg, rgba(255, 253, 250, 0.98), rgba(250, 246, 241, 0.98));
    border: 1px solid var(--ci-border);
    border-radius: 10px;
    box-shadow: var(--ci-shadow);
    padding: 1.25rem 1.35rem;
    margin-bottom: 1rem;
}

.cc-pdf-capture-screen h1 {
    margin: 0 !important;
}

.cc-pdf-capture-screen p:not(.cc-eyebrow) {
    margin: 0.55rem 0 0;
    color: var(--ci-muted);
}

@media (max-height: 760px) {
    .cc-sidebar-status,
    .sidebar-brand {
        display: none !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        min-height: 36px !important;
        font-size: 0.9rem !important;
        padding-top: 0.42rem !important;
        padding-bottom: 0.42rem !important;
    }
}

@media (max-width: 1180px) {
    .cc-hero-panel {
        grid-template-columns: 1fr;
    }
    .cc-summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .cc-station-intro,
    .cc-map-heading {
        align-items: flex-start;
        flex-direction: column;
    }
}


/* Inline Info Tooltips */
.cc-info-tip {
    position: relative;
    display: inline-flex;
    align-items: center;
    vertical-align: middle;
    margin-left: 4px;
}
.cc-info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    font-size: 12px;
    color: #c0825a;
    cursor: help;
    opacity: 0.7;
    transition: opacity 0.2s ease, transform 0.2s ease;
    border-radius: 50%;
}
.cc-info-icon:hover,
.cc-info-icon:focus {
    opacity: 1;
    transform: scale(1.15);
}
.cc-info-popup {
    display: none;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255, 253, 250, 0.98);
    color: #3d2e22;
    font-size: 0.78rem;
    font-weight: 400;
    line-height: 1.5;
    padding: 10px 14px;
    border-radius: 10px;
    border: 1px solid rgba(139, 90, 43, 0.12);
    box-shadow: 0 8px 28px rgba(100, 60, 20, 0.12);
    min-width: 220px;
    max-width: 320px;
    z-index: 9999;
    white-space: normal;
    pointer-events: none;
    animation: cc-tip-in 0.2s ease;
}
.cc-info-tip:hover .cc-info-popup,
.cc-info-icon:focus + .cc-info-popup {
    display: block;
}
@keyframes cc-tip-in {
    from { opacity: 0; transform: translateX(-50%) translateY(4px); }
    to   { opacity: 1; transform: translateX(-50%) translateY(0); }
}

/* Key Takeaways Narrative Card */
.cc-key-takeaways {
    background: linear-gradient(135deg, rgba(192, 130, 90, 0.06), rgba(181, 148, 94, 0.05));
    border: 1px solid rgba(192, 130, 90, 0.15);
    border-radius: 16px;
    padding: 24px 28px;
    margin: 16px 0 24px 0;
}
.cc-key-takeaways h4 {
    margin: 0 0 14px 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #3d2e22;
    letter-spacing: -0.01em;
}
.cc-key-takeaways ul {
    list-style: none;
    padding: 0;
    margin: 0;
}
.cc-key-takeaways li {
    position: relative;
    padding: 6px 0 6px 28px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #5a4435;
}
.cc-key-takeaways li::before {
    content: "\\2728";
    position: absolute;
    left: 0;
    top: 6px;
    font-size: 1rem;
}

/* Metric Descriptor Subtitle */
.cc-metric-descriptor {
    display: block;
    font-size: 0.75rem;
    color: #a89585;
    font-weight: 400;
    margin-top: 2px;
    letter-spacing: 0.01em;
}

/* ── Smooth scrollbar for the whole app ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(192, 130, 90, 0.2);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(192, 130, 90, 0.35);
}

/* ── Selection color ── */
::selection {
    background: rgba(192, 130, 90, 0.2);
    color: #3d2e22;
}
</style>
'''
st.markdown(CLIMATE_INTELLIGENCE_CSS, unsafe_allow_html=True)
