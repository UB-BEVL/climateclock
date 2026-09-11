# Psychrometric Studio integration

Adapted from https://github.com/patpease/psychrometric-studio at commit
`e5d897b32c992ff1cb6b0af6d34ba03786adaa6a` (MIT, Copyright 2026 Pease Studio).
Original licensing is in LICENSE; bundled dependency notices are in
build/third-party-notices.txt and source/THIRD-PARTY-NOTICES.md.

## Use

Open Detail View > Psychrometrics > Interactive chart. Your loaded station is
connected automatically. The studio includes all 17 equipment types, separate
operating cases, SI/IP conversion, chart lines, pan/zoom, draggable source points,
PMV/adaptive comfort, weather density/scatter and filters, EPW/ZIP/DDY loading,
teaching references, the cooling-coil walkthrough, and project/chart exports.
Focus chart temporarily hides the controls. Climate Strategies retains the
existing passive-design analysis.

Design and weather edits persist in this browser tab while navigating the app.
Reset Session starts a fresh station and design. Save a JSON project for longer
term storage; a browser reload starts a new Streamlit session. Share links contain
the project definition but do not carry the weather file.

Wait for the studio's report status to say the operating cases are included
before opening Report. The full climate PDF includes each case's chart and solved
state/duty tables. The PDF controls beneath the chart export all cases or an
individual case, with the BEVL and UB logos. Full-report charts use a landscape
view that includes all design points and selected weather. Python
creates the PDFs from the browser's solved results; no separate report API is
needed. Weather URL fetching is limited to HTTPS ZIPs on climate.onebuilding.org.
Uploaded EPW files normally do not include DDY data; upload an archive with DDY
data in the studio when you want design-day markers.

## Edit and rebuild

The editable React/TypeScript code is in source/web/src. ClimateClock-specific
transport, initial-unit conversion, and report synchronization are in
source/web/src/climateclock; the main layout is ui/App.tsx and ui/styles.css.
The Python component and PDF layout are in __init__.py.

From source/web, using a current Node.js version supported by Vite:

```powershell
npm ci
npm run build
npm test -- --maxWorkers=2
Copy-Item -Path dist/* -Destination ../../build -Recurse -Force
```

Restart the Streamlit app after replacing the bundle. Node.js is only needed to
edit/rebuild the frontend; running the app uses the included build.

Integration fixes also make weather filters affect the displayed/exported
weather layer, convert restored weather when project units change, serialize
Streamlit requests, and prevent report timestamps triggering repeated updates.
The upstream standalone API, deployment worker, feedback endpoint, and hosting
configuration are not used by this integration.
