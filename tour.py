"""A non-blocking, action-aware tour that survives Streamlit reruns."""
from __future__ import annotations

import json
from typing import Any, Iterable

import streamlit.components.v1 as components


def run_onboarding_tour(
    steps: Iterable[dict[str, Any]],
    *,
    key: str,
    stage: str,
    delay_ms: int = 250,
) -> None:
    """Render every run; the parent browser owns progress and dismissal.

    The station stage waits for a successful load. Each analysis page keeps
    independent Next/Back progress. A new session key restarts all guides.
    """
    if stage not in {"station", "overview", "detail", "report", "paused"} and not stage.startswith("detail:"):
        raise ValueError(f"Unknown onboarding stage: {stage}")
    payload = json.dumps({
        "key": key, "stage": stage, "delayMs": max(0, int(delay_ms)),
        "steps": [dict(step) for step in steps],
    }).replace("<", "\\u003c")
    parent_script = _PARENT_SCRIPT.replace("__TOUR_CONFIG__", payload)
    # Inline components are replaced on reruns. Create the runtime in the main
    # document so its event handlers and timers outlive the component iframe.
    encoded_script = json.dumps(parent_script).replace("<", "\\u003c")
    components.html(_TOUR_HTML.replace("__PARENT_SCRIPT__", encoded_script), height=0)


_TOUR_HTML = r'''
<script>
(() => {
  const script = window.parent.document.createElement("script");
  script.textContent = __PARENT_SCRIPT__;
  window.parent.document.head.appendChild(script);
  script.remove();
})();
</script>
'''


_PARENT_SCRIPT = r'''
(() => {
  const config = __TOUR_CONFIG__;
  const host = window;
  const doc = host.document;
  const runtimeKey = "__climateClockGuidedTourV2";

  // Replace the old single-page runtime when Streamlit hot-reloads this file.
  const previous = host[runtimeKey];
  if (previous && previous.version !== 8) {
    if (previous.timer) host.clearInterval(previous.timer);
    previous.hide();
    for (const id of ["cc-guided-tour", "cc-guided-tour-ring", "cc-guided-tour-style"]) {
      const element = doc.getElementById(id);
      if (element) element.remove();
    }
    delete host[runtimeKey];
  }

  if (!host[runtimeKey]) {
    const style = doc.createElement("style");
    style.id = "cc-guided-tour-style";
    style.textContent = `
      #cc-guided-tour {
        position: fixed; z-index: 1000002; width: 330px;
        max-width: calc(100vw - 24px); max-height: calc(100vh - 24px);
        overflow: auto; box-sizing: border-box; padding: 22px;
        border: 1px solid #dac7b7; border-radius: 16px;
        background: #fffdf9; color: #3d2e22;
        box-shadow: 0 16px 48px #3d2e222e;
        font: 14px/1.55 Inter, system-ui, sans-serif;
        text-align: left; color-scheme: light;
      }
      #cc-guided-tour[hidden], #cc-guided-tour-ring[hidden] { display: none; }
      #cc-guided-tour .cc-gt-kicker {
        margin: 0 28px 10px 0; color: #9a6541;
        font-size: 10px; font-weight: 800; letter-spacing: 1.4px;
        text-transform: uppercase;
      }
      #cc-guided-tour h2 { margin: 0 20px 10px 0; font-size: 19px; line-height: 1.25; color: #3d2e22; }
      #cc-guided-tour p { margin: 0; color: #6a5746; font-size: 14px; }
      #cc-guided-tour .cc-gt-status {
        margin-top: 14px; padding: 9px 11px; border-radius: 8px;
        background: #f1e8db; color: #705236; font-size: 12px;
      }
      #cc-guided-tour .cc-gt-footer { display: flex; align-items: center; gap: 8px; margin-top: 20px; }
      #cc-guided-tour button {
        appearance: none; border: 1px solid #dac7b7; border-radius: 8px;
        padding: 8px 12px; background: transparent; color: #5d4736;
        font: 600 12px/1.4 Inter, system-ui, sans-serif; cursor: pointer;
      }
      #cc-guided-tour button:focus-visible { outline: 3px solid #c0825a; outline-offset: 3px; }
      #cc-guided-tour button:hover { background: #f1e8db; }
      #cc-guided-tour button.cc-gt-primary { background: #925c3b; color: white; border-color: #925c3b; }
      #cc-guided-tour button.cc-gt-primary:hover { background: #75482e; }
      #cc-guided-tour .cc-gt-skip { border: 0; padding-left: 0; margin-right: auto; color: #806d5d; }
      #cc-guided-tour .cc-gt-close { position: absolute; right: 12px; top: 12px; border: 0; padding: 3px 8px; font-size: 20px; }
      #cc-guided-tour-ring {
        position: fixed; z-index: 1000001; pointer-events: none;
        border: 2px solid #c0825a; border-radius: 10px;
        box-shadow: 0 0 0 4px #c0825a20; box-sizing: border-box;
      }
      @media (prefers-reduced-motion: no-preference) {
        #cc-guided-tour { transition: top .12s ease, left .12s ease; }
      }
    `;
    doc.head.appendChild(style);
    const card = doc.createElement("section");
    card.id = "cc-guided-tour";
    card.hidden = true;
    card.setAttribute("role", "dialog");
    card.setAttribute("aria-modal", "false");
    card.setAttribute("aria-labelledby", "cc-gt-title");
    card.setAttribute("aria-describedby", "cc-gt-description");
    const ring = doc.createElement("div");
    ring.id = "cc-guided-tour-ring";
    ring.hidden = true;
    ring.setAttribute("aria-hidden", "true");
    doc.body.append(ring, card);

    const runtime = {
      version: 8, progress: {}, scope: null, steps: [],
      config: null, key: null, stage: null, index: 0,
      dismissed: false, completed: false, target: null,
      renderedStep: null, scrollPending: true, missingSince: 0, timer: null,
      isVisible(element) {
        const rect = element.getBoundingClientRect();
        const css = host.getComputedStyle(element);
        // Children of closed <details> can retain nonzero layout rectangles.
        // Only its summary is a visible guide target until the user opens it.
        for (let parent = element; parent; parent = parent.parentElement) {
          const style = host.getComputedStyle(parent);
          if (style.display === "none" || style.visibility === "hidden" || style.contentVisibility === "hidden") return false;
          if (parent.tagName === "DETAILS" && !parent.open) {
            const summary = parent.querySelector("summary");
            if (!summary || !summary.contains(element)) return false;
          }
        }
        return element.isConnected && rect.width > 0 && rect.height > 0
          && css.visibility !== "hidden" && css.display !== "none"
          && !element.closest("[hidden], [aria-hidden='true']");
      },
      text(element) { return (element?.textContent || "").replace(/\s+/g, " ").trim(); },
      explain(kind, title) {
        const low = title.toLowerCase();
        if (kind === "charts") {
          if (/psychrom/.test(low)) return "Each plotted air condition combines temperature and moisture. Compare the hourly points with the comfort region and strategy overlays to see which conditions dominate.";
          if (/wind.*rose|rose.*wind/.test(low)) return "The sectors show where wind comes from. Longer sectors mean more frequent winds; colours distinguish wind-speed ranges. Check the legend and calm-hours note.";
          if (/sun.*path|cartesian|analemma/.test(low)) return "This traces the sun's position at different hours and seasons. Read the altitude and azimuth labels, and use the date or orientation controls to explore solar access and shading.";
          if (/heatmap|heat map|diurnal resource/.test(low)) return "Colour represents the value or stress category for each time interval. Read the day/month and hour axes, then compare colours with the legend to find recurring daily and seasonal patterns.";
          if (/degree|heating|cooling|load/.test(low)) return "These temperature-based indicators summarize heating or cooling demand relative to a base temperature. Compare the periods and base-temperature settings shown; these are climate indicators, not a building energy simulation.";
          if (/comfort|stress|utci|pmv|discomfort/.test(low)) return "Compare comfortable periods with heat- and cold-stress conditions. Read the model, thresholds, and valid-data notes alongside the legend before interpreting the percentages or hours.";
          if (/precip|rain|snow/.test(low)) return "Compare wet and dry periods using the totals or frequencies shown on the axes. Check the weather source and missing-data notes, especially when the EPW file has incomplete precipitation records.";
          if (/scatter|dot|distribution|box|histogram/.test(low)) return "The spread shows how much hourly conditions vary. Look for the dense central range, extremes, and seasonal differences; use the axes and legend to identify each group.";
          if (/solar|irradiance|insolation|ghi|dni|dhi/.test(low)) return "Compare the available solar energy across the displayed periods. GHI is radiation on a horizontal surface; DNI is direct-beam radiation, and DHI is diffuse-sky radiation. Read the units carefully.";
          if (/monthly|annual|temperature|humidity|wind|cloud/.test(low)) return "Compare the plotted climate conditions across the months or hours shown. Look for peaks, low periods, and seasonal shifts, using the axis labels and legend to distinguish the series.";
          return "Read the axis labels and legend to identify the values and periods being compared. Hover over the chart for exact values and compare its pattern with the other charts in this subtab.";
        }
        if (kind === "metrics") {
          if (/humidity|rh/.test(low)) return "Relative humidity measures how moist the air is compared with what it can hold at that temperature. This card summarizes the data using the statistic in its label.";
          if (/wind/.test(low)) return "This summarizes wind conditions for the loaded weather file. Read the speed units and compare it with the wind charts for direction, variability, and calmer periods.";
          if (/solar|ghi|radiation/.test(low)) return "This summarizes the solar resource. Check whether the units describe average power (W/m²) or accumulated energy (kWh/m²).";
          if (/comfort|stress|di |utci|pmv/.test(low)) return "This summarizes comfort or thermal stress for the model and thresholds shown. Check whether the value is hours or a percentage, and read any data-availability or occupancy notes.";
          if (/temp|dry|hot|cold|focus/.test(low)) return "This summarizes air temperature or the hours crossing the threshold in the label. Check the temperature units and compare the annual summary with the seasonal charts.";
          return "This is a summary statistic for the loaded weather data. The label gives the measure, while the value and any note underneath give its units and coverage.";
        }
        if (kind === "settings") return "Open this panel to adjust the settings for this analysis. The surrounding charts update when you change a value, and the guide keeps your place.";
        if (kind === "controls") return "Use this control to change the period, statistic, or analysis option shown in its label. The page reruns with your choice, and the guide keeps your place.";
        if (kind === "tables") return "Inspect the underlying values, field names, and units here. Completeness tables show missing data; weather tables let you check the records behind the charts.";
        return "Use this button to save the chart or data in the format shown. For the complete climate report as one PDF, open Report in the sidebar.";
      },
      collectSteps(selectedTabs) {
        const result = [];
        this.config.steps.forEach((base, baseIndex) => {
          if (!base.expand) {
            const step = {...base, id: `step-${baseIndex}`};
            if (base.subtab_intro && selectedTabs.length) {
              const label = this.text(selectedTabs[selectedTabs.length - 1]);
              const copy = base.subtab_descriptions || {};
              step.element = selectedTabs[selectedTabs.length - 1];
              step.title = `${label} guide`;
              step.description = copy[label] || base.description;
            }
            if (!base.optional || !this.stage.startsWith("detail:") || this.findTarget(step)) result.push(step);
            return;
          }
          const elements = [...doc.querySelectorAll(base.selector)].filter(el => this.isVisible(el));
          elements.forEach((element, itemIndex) => {
            let title;
            if (base.expand === "charts") {
              const headings = [...doc.querySelectorAll(".st-key-tour_detail_content :is(h1,h2,h3,h4,h5)")]
                .filter(heading => this.isVisible(heading) && (heading.compareDocumentPosition(element) & 4));
              title = this.text(element.querySelector(".gtitle")) || this.text(headings[headings.length - 1])
                || `${this.stage.split(":")[1]} chart ${itemIndex + 1}`;
            } else if (base.expand === "metrics") {
              title = this.text(element.querySelector("[data-testid='stMetricLabel']")) || "Summary statistic";
            } else if (base.expand === "tables") {
              title = this.stage.includes("Raw Data") ? "Data quality and hourly records" : "Read the data table";
            } else {
              title = this.text(element.querySelector("label")) || this.text(element);
            }
            result.push({element, title, description: this.explain(base.expand, title),
              id: `${baseIndex}-${itemIndex}-${title}`, optional: true});
          });
        });
        return result;
      },
      syncContext() {
        const tabs = this.stage.startsWith("detail:")
          ? [...doc.querySelectorAll(".st-key-tour_detail_content [role='tab'][aria-selected='true']")].filter(el => this.isVisible(el)) : [];
        const scope = this.stage + (tabs.length ? "::" + tabs.map(el => this.text(el)).join("::") : "");
        const changed = scope !== this.scope;
        const previousStep = changed ? null : this.steps[this.index]?.id;
        if (changed) {
          this.saveProgress();
          this.scope = scope;
          const saved = this.progress[scope] || {};
          this.index = saved.index || 0;
          this.dismissed = saved.dismissed || false;
          this.completed = saved.completed || false;
          this.renderedStep = null;
          this.scrollPending = true;
          this.missingSince = 0;
        }
        this.steps = this.collectSteps(tabs);
        if (previousStep) {
          const position = this.steps.findIndex(step => step.id === previousStep);
          if (position >= 0) this.index = position;
        }
        this.index = Math.min(this.index, Math.max(0, this.steps.length - 1));
      },
      hide() { card.hidden = true; ring.hidden = true; },
      saveProgress() {
        if (this.scope && this.stage !== "paused") {
          this.progress[this.scope] = {
            index: this.index, dismissed: this.dismissed, completed: this.completed,
          };
        }
      },
      finish(skipped) {
        this.dismissed = skipped;
        this.completed = !skipped;
        this.saveProgress();
        this.hide();
        // Nested Streamlit tabs switch without a Python rerun. Keep watching
        // them after Done/Skip so the next subtab can start its own guide.
        if (!this.stage.startsWith("detail:")) {
          if (this.timer) host.clearInterval(this.timer);
          this.timer = null;
        }
        if (this.target && this.target.isConnected) {
          const focusTarget = this.target.matches("button, input, a")
            ? this.target : this.target.querySelector("button, input, a");
          if (focusTarget) focusTarget.focus({preventScroll: true});
        }
      },
      findTarget(step) {
        if (step.element) return this.isVisible(step.element) ? step.element : null;
        for (const selector of [step.selector, ...(step.fallback_selectors || [])]) {
          if (!selector) continue;
          for (const element of doc.querySelectorAll(selector)) {
            if (selector === step.selector && step.match_text) {
              const text = element.textContent.replace(/\s+/g, " ").trim();
              if (step.match_mode === "contains" ? !text.includes(step.match_text) : text !== step.match_text) continue;
            }
            const rect = element.getBoundingClientRect();
            if (element.closest("[data-testid='stSidebar']") && (rect.right <= 0 || rect.left >= doc.documentElement.clientWidth)) continue;
            if (this.isVisible(element)) return element;
          }
        }
        return null;
      },
      move(delta) {
        const step = this.steps[this.index];
        // Only the app's successful load can advance the required action.
        if (delta > 0 && step.wait_for_station) return;
        const next = this.index + delta;
        if (next >= this.steps.length) { this.finish(false); return; }
        this.index = Math.max(0, next);
        this.saveProgress();
        this.renderedStep = null;
        this.scrollPending = true;
        this.missingSince = 0;
        this.tick();
        const button = card.querySelector(".cc-gt-primary") || card.querySelector(".cc-gt-back");
        if (button) button.focus({preventScroll: true});
      },
      render(step) {
        card.replaceChildren();
        const add = (tag, className, content) => {
          const node = doc.createElement(tag);
          node.className = className;
          node.textContent = content;
          card.appendChild(node);
          return node;
        };
        const close = add("button", "cc-gt-close", "×");
        close.setAttribute("aria-label", "Skip this page's guide");
        close.onclick = () => this.finish(true);
        const labels = {station: "Getting started", overview: "Overview guide", detail: "Detail View guide", report: "Report guide"};
        add("div", "cc-gt-kicker", `${labels[this.stage.split(":")[0]]} · ${this.index + 1} of ${this.steps.length}`);
        const title = add("h2", "", step.title);
        title.id = "cc-gt-title";
        const description = add("p", "", step.description);
        description.id = "cc-gt-description";
        if (step.wait_for_station) add("p", "cc-gt-status", "The tour continues automatically once your station loads.");
        const footer = add("div", "cc-gt-footer", "");
        const button = (text, className, action) => {
          const node = doc.createElement("button");
          node.type = "button";
          node.className = className;
          node.textContent = text;
          node.onclick = action;
          footer.appendChild(node);
        };
        button("Skip guide", "cc-gt-skip", () => this.finish(true));
        if (this.index > 0) button("Back", "cc-gt-back", () => this.move(-1));
        if (!step.wait_for_station) button(step.next_label || (this.index === this.steps.length - 1 ? "Done" : "Next"), "cc-gt-primary", () => this.move(1));
        card.onkeydown = event => {
          if (event.key === "Escape") { event.stopPropagation(); this.finish(true); }
        };
        this.renderedStep = `${this.scope}:${this.index}:${this.steps.length}`;
      },
      position() {
        if (card.hidden || !this.target) return;
        const rect = this.target.getBoundingClientRect();
        const vw = doc.documentElement.clientWidth, vh = host.innerHeight;
        const pad = 12, gap = 16, width = card.offsetWidth, height = card.offsetHeight;
        const clamp = (value, low, high) => Math.max(low, Math.min(value, Math.max(low, high)));
        let x = rect.right + gap, y = rect.top;
        if (x + width > vw - pad) {
          x = rect.left - width - gap;
          if (x < pad) {
            x = rect.right - width;
            y = rect.bottom + gap;
            if (y + height > vh - pad) y = rect.top - height - gap;
          }
        }
        // Keep neighbouring section buttons and nested tabs clickable.
        const navigation = this.target.closest(".st-key-dashboard_section_nav, [role='tablist']");
        if (navigation) {
          const navRect = navigation.getBoundingClientRect();
          x = rect.left;
          y = navRect.bottom + gap;
          if (y + height > vh - pad && navRect.top - height - gap >= pad) y = navRect.top - height - gap;
        }
        x = clamp(x, pad, vw - width - pad);
        for (const bar of doc.querySelectorAll(".st-key-dashboard_section_nav, .st-key-tour_detail_content [role='tablist']")) {
          const bounds = bar.getBoundingClientRect();
          if (bounds.bottom <= 0 || bounds.top >= vh || !this.isVisible(bar)) continue;
          if (x < bounds.right && x + width > bounds.left && y < bounds.bottom && y + height > bounds.top) {
            if (bounds.bottom + gap + height <= vh - pad) y = bounds.bottom + gap;
            else if (bounds.top - height - gap >= pad) y = bounds.top - height - gap;
          }
        }
        card.style.left = `${clamp(x, pad, vw - width - pad)}px`;
        card.style.top = `${clamp(y, pad, vh - height - pad)}px`;
        const top = Math.max(rect.top - 5, 0), left = Math.max(rect.left - 5, 0);
        const right = Math.min(rect.right + 5, vw), bottom = Math.min(rect.bottom + 5, vh);
        ring.hidden = right <= left || bottom <= top;
        Object.assign(ring.style, {left: `${left}px`, top: `${top}px`, width: `${Math.max(0, right - left)}px`, height: `${Math.max(0, bottom - top)}px`});
      },
      tick() {
        if (this.config.stage === "paused") { this.hide(); return; }
        this.syncContext();
        if (this.dismissed || this.completed) { this.hide(); return; }
        const step = this.steps[this.index];
        if (!step) { this.hide(); return; }
        const target = this.findTarget(step);
        if (!target) {
          this.hide();
          if (!this.missingSince) this.missingSince = Date.now();
          // Sparse weather files may omit optional charts.
          if (step.optional && Date.now() - this.missingSince > 8000) this.move(1);
          return;
        }
        this.missingSince = 0;
        this.target = target;
        if (this.renderedStep !== `${this.scope}:${this.index}:${this.steps.length}`) this.render(step);
        card.hidden = false;
        if (this.scrollPending) {
          target.scrollIntoView({block: "center", inline: "nearest", behavior: "instant"});
          this.scrollPending = false;
        }
        this.position();
      },
      update(next) {
        const fresh = next.key !== this.key;
        this.saveProgress();
        if (fresh) {
          this.key = next.key;
          this.stage = null;
          this.scope = null;
          this.progress = {};
          this.index = 0;
          this.dismissed = false;
          this.completed = false;
        }
        if (next.stage !== this.stage || fresh) {
          this.stage = next.stage;
          this.scope = null;
          this.renderedStep = null;
          this.scrollPending = true;
          this.missingSince = 0;
          this.target = null;
        }
        this.config = next;
        if (this.timer) host.clearInterval(this.timer);
        this.timer = null;
        this.hide();
        if (next.stage !== "paused") {
          this.timer = host.setInterval(() => this.tick(), 300);
          host.setTimeout(() => this.tick(), next.delayMs);
        }
      },
    };
    host.addEventListener("resize", () => runtime.position());
    doc.addEventListener("scroll", () => runtime.position(), true);
    host[runtimeKey] = runtime;
  }
  host[runtimeKey].update(config);
})();
'''
