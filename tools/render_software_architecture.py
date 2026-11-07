from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def box(ax, xy, width, height, title, detail, *, face, edge="#253247", title_size=9.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.05, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height * 0.64, title,
        ha="center", va="center", fontsize=title_size, fontweight="bold",
        color="#162033", zorder=3,
    )
    ax.text(
        x + width / 2, y + height * 0.31, detail,
        ha="center", va="center", fontsize=7.4, color="#38475f",
        linespacing=1.15, zorder=3,
    )
    return patch


def arrow(ax, start, end, *, color="#66758c", width=1.15, bend=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9.5,
        linewidth=width, color=color,
        connectionstyle=f"arc3,rad={bend}", shrinkA=3, shrinkB=3, zorder=1,
    ))


def main():
    fig, ax = plt.subplots(figsize=(12.0, 7.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.965, "Climate Analysis Pro: Software Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#162033",
    )
    ax.text(
        0.5, 0.928,
        "Weather ingestion, analysis services, future-weather generation, and research outputs",
        ha="center", va="center", fontsize=9, color="#526176",
    )

    # Column headings
    headings = [(0.095, "DATA SOURCES"), (0.31, "APPLICATION"), (0.645, "ANALYSIS MODULES"), (0.915, "OUTPUTS")]
    for x, label in headings:
        ax.text(x, 0.875, label, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#40506a")

    input_face = "#EAF2F8"
    app_face = "#E8F3EE"
    module_face = "#F2EFF8"
    output_face = "#FBF1E5"

    # Inputs
    source_boxes = [
        box(ax, (0.018, 0.695), 0.155, 0.105, "EPW weather files", "Station catalogue\nor user upload", face=input_face),
        box(ax, (0.018, 0.535), 0.155, 0.105, "Sensor observations", "CSV upload or\nlive API", face=input_face),
        box(ax, (0.018, 0.375), 0.155, 0.105, "Forecast service", "Open-Meteo\n10-day forecast", face=input_face),
        box(ax, (0.018, 0.215), 0.155, 0.105, "Climate projections", "CMIP6 monthly deltas\nSSP1-2.6 to SSP5-8.5", face=input_face),
    ]

    # Main app and shared data state
    app = box(
        ax, (0.215, 0.565), 0.205, 0.20,
        "Streamlit application",
        "app.py\nRouting, session state, data validation,\nvisualization, and workflow orchestration",
        face=app_face, title_size=10.5,
    )
    state = box(
        ax, (0.230, 0.355), 0.175, 0.105,
        "Canonical hourly dataset",
        "Pandas DataFrame + EPW metadata\nshared through session state",
        face="#F5F7F8",
    )

    # Analysis modules
    module_specs = [
        (0.470, 0.715, "Short-term\nforecasting", "models/forecasting.py\nForecast, bias, extremes, degree days"),
        (0.655, 0.715, "Future-weather\nmorphing", "models/future_epw.py\nSensor bias + CMIP6 deltas + EPW export"),
        (0.470, 0.535, "Sensor processing", "live_sensors.py\nsensors/preprocess.py\nIngest, normalize, climatology"),
        (0.655, 0.535, "Comfort and loads", "metrics/comfort_energy.py\nUTCI, PMV, DI, adaptive comfort"),
        (0.470, 0.355, "Psychrometrics", "psychro_helpers.py\nAir-state properties and strategy zones"),
        (0.655, 0.355, "Reporting", "pdf_chart_renderer.py\nChart rendering and PDF assembly"),
    ]
    module_boxes = [
        box(ax, (x, y), 0.165, 0.125, title, detail, face=module_face, title_size=8.6)
        for x, y, title, detail in module_specs
    ]

    # Outputs
    output_specs = [
        (0.850, 0.710, "Interactive\ndashboards", "Climate, comfort, solar,\npsychrometric, and wind views"),
        (0.850, 0.535, "Forecast products", "10-day outlook and\nEPW-baseline comparison"),
        (0.850, 0.360, "Morphed EPW files", "2050 and 2080 outputs\nfor selected SSP pathway"),
        (0.850, 0.185, "Research exports", "PDF report, figures, tables,\nand GIS-ready datasets"),
    ]
    output_boxes = [
        box(ax, (x, y), 0.135, 0.125, title, detail, face=output_face, title_size=8.0)
        for x, y, title, detail in output_specs
    ]

    # Data flow into app/state
    arrow(ax, (0.173, 0.748), (0.215, 0.695))
    arrow(ax, (0.173, 0.588), (0.215, 0.635))
    arrow(ax, (0.173, 0.428), (0.470, 0.777), bend=-0.18)
    arrow(ax, (0.173, 0.268), (0.655, 0.777), bend=-0.11)
    arrow(ax, (0.318, 0.565), (0.318, 0.460), color="#358167", width=1.4)

    # Shared state to modules; app also controls report assembly.
    module_left = [(0.470, 0.777), (0.655, 0.777), (0.470, 0.598),
                   (0.655, 0.598), (0.470, 0.418), (0.655, 0.418)]
    starts = [(0.405, 0.410), (0.405, 0.425), (0.405, 0.405),
              (0.405, 0.420), (0.405, 0.395), (0.405, 0.410)]
    bends = [-0.18, -0.12, -0.08, -0.04, 0.02, 0.08]
    for start, end, bend in zip(starts, module_left, bends):
        arrow(ax, start, end, bend=bend)

    # Selected module interactions.
    arrow(ax, (0.635, 0.777), (0.655, 0.777), color="#7B63A6", width=1.0)
    arrow(ax, (0.635, 0.598), (0.655, 0.598), color="#7B63A6", width=1.0)
    arrow(ax, (0.635, 0.418), (0.655, 0.418), color="#7B63A6", width=1.0)

    # Outputs
    arrow(ax, (0.420, 0.665), (0.850, 0.772), bend=-0.08)
    arrow(ax, (0.635, 0.777), (0.850, 0.598))
    arrow(ax, (0.820, 0.777), (0.850, 0.423))
    arrow(ax, (0.820, 0.418), (0.850, 0.248))

    # Flow legend
    ax.plot([0.035, 0.065], [0.085, 0.085], color="#66758c", lw=1.3)
    ax.text(0.072, 0.085, "Data flow", va="center", fontsize=7.5, color="#526176")
    ax.plot([0.155, 0.185], [0.085, 0.085], color="#7B63A6", lw=1.3)
    ax.text(0.192, 0.085, "Module dependency", va="center", fontsize=7.5, color="#526176")
    ax.plot([0.315, 0.345], [0.085, 0.085], color="#358167", lw=1.5)
    ax.text(0.352, 0.085, "Shared application state", va="center", fontsize=7.5, color="#526176")

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(OUT / f"software-architecture.{suffix}", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "software-architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
