import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xml.etree.ElementTree as ET
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Optional basemap support
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except Exception:
    HAS_CONTEXTILY = False

# -----------------------------
# Configuration
# -----------------------------
EXCEL_PATH = r"C:\Users\jt00784\Desktop\GPS Data\GPS Data For LPF.xlsx"
SHEET_NAME = "GPS Data For LPF"  # fallback to first sheet if not found
KML_PATH = r"C:\Users\jt00784\Desktop\GPS Data\Fault.kml"  # KML titled "Fault"
OUTPUT_PNG = r"C:\Users\jt00784\Desktop\GPS Data\Output\gps_velocity_vector_map.png"

SHOW_LABELS = True
USE_BASEMAP = True
BASEMAP_ZOOM = 12

# Arrow visibility
TARGET_ARROW_FRACTION_OF_MAP = 0.08  # median arrow length ~8% of map span

# Fixed view
USE_FIXED_VIEW = True
VIEW_WIDTH_KM = 25
VIEW_HEIGHT_KM = 20

# Scale bar
SCALE_BAR_KM = 5

# Ignore these stations
EXCLUDED_STATIONS = {"CSTA", "TSFT"}

# Label offset tweaks (meters in EPSG:3857)
LABEL_OFFSETS_M = {
    "HCC2": (400.0, -400.0),  # move HCC2 down so it does not cover HCC1
}


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_col(df: pd.DataFrame, exact_names, contains_names=None):
    cols = list(df.columns)
    for n in exact_names:
        if n in cols:
            return n
    if contains_names:
        for c in cols:
            cl = c.lower()
            for n in contains_names:
                if n.lower() in cl:
                    return c
    return None


def load_excel(path: str, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError:
        return pd.read_excel(path, engine="openpyxl")
    except ImportError:
        raise SystemExit(
            "Missing dependency: openpyxl\n"
            "Install with: py -3.14 -m pip install openpyxl"
        )


def parse_kml_fault_lines(kml_path: str):
    """Read all LineString coordinates from a KML."""
    if not os.path.exists(kml_path):
        print(f"KML not found: {kml_path}")
        return []

    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        coord_nodes = root.findall(".//{*}LineString/{*}coordinates")
        lines = []

        for node in coord_nodes:
            if node.text is None:
                continue
            coord_text = node.text.strip()
            if not coord_text:
                continue

            pts = []
            for token in coord_text.split():
                parts = token.split(",")
                if len(parts) < 2:
                    continue
                lon = float(parts[0])
                lat = float(parts[1])
                pts.append([lon, lat])

            if len(pts) >= 2:
                lines.append(np.array(pts, dtype=float))

        return lines

    except Exception as e:
        print(f"Failed to parse KML ({kml_path}): {e}")
        return []


def add_distance_scale(ax, length_km=5):
    """Draw a simple distance scale bar in map units (EPSG:3857 meters)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0

    length_m = length_km * 1000.0
    sx = x0 + 0.04 * dx
    sy = y0 + 0.06 * dy
    ex = sx + length_m

    ax.plot([sx, ex], [sy, sy], color="white", lw=5, zorder=10)
    ax.plot([sx, ex], [sy, sy], color="black", lw=2, zorder=11)
    ax.plot([sx, sx], [sy - 0.008 * dy, sy + 0.008 * dy], color="black", lw=2, zorder=11)
    ax.plot([ex, ex], [sy - 0.008 * dy, sy + 0.008 * dy], color="black", lw=2, zorder=11)
    txt = ax.text(
        (sx + ex) / 2, sy + 0.012 * dy, f"{length_km} km",
        ha="center", va="bottom", fontsize=10, color="black", zorder=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2)
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def add_north_arrow(ax):
    """Add high-visibility north arrow, placed below legend."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0

    # moved a bit right
    x = x0 + 0.945 * dx
    y_start = y0 + 0.60 * dy
    y_end = y0 + 0.75 * dy

    # white backing panel
    ax.text(
        x, y0 + 0.76 * dy, " ",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.75, boxstyle="round,pad=0.35"),
        zorder=19
    )

    # halo + foreground arrow
    ax.annotate(
        "", xy=(x, y_end), xytext=(x, y_start),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=6.0),
        zorder=20
    )
    ax.annotate(
        "", xy=(x, y_end), xytext=(x, y_start),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.6),
        zorder=21
    )

    t = ax.text(
        x, y_end + 0.015 * dy, "N",
        ha="center", va="bottom", fontsize=16, fontweight="bold",
        color="black", zorder=22
    )
    t.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])


def apply_lonlat_axes(ax):
    """Show axis ticks as lon/lat while data remains EPSG:3857."""
    R = 6378137.0

    def x_to_lon(x, pos):
        lon = (x / R) * (180.0 / np.pi)
        return f"{lon:.3f}°"

    def y_to_lat(y, pos):
        lat = (2.0 * np.arctan(np.exp(y / R)) - np.pi / 2.0) * (180.0 / np.pi)
        return f"{lat:.3f}°"

    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(FuncFormatter(x_to_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(y_to_lat))

    ax.tick_params(axis="both", labelsize=10, colors="black", width=1.2)
    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])


def main():
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)

    df = load_excel(EXCEL_PATH, SHEET_NAME)
    df = normalize_cols(df)

    # Expected headers
    station_col = get_col(df, ["Station:"], ["station"])
    lat_col = get_col(df, ["Lat:"], ["lat"])
    lon_col = get_col(df, ["Lon:"], ["lon", "longitude"])
    vn_col = get_col(df, ["North-South Rates(mm):"], ["north-south rates(mm)", "north", "ns"])
    ve_col = get_col(df, ["East-West Rates(mm):"], ["east-west rates(mm)", "east", "ew"])
    fault_col = get_col(df, ["Fault Position:"], ["fault position", "fault"])

    required = [lat_col, lon_col, vn_col, ve_col]
    if any(c is None for c in required):
        raise ValueError(
            f"Required columns not found.\n"
            f"Columns detected: {list(df.columns)}\n"
            f"Need: Lat:, Lon:, North-South Rates(mm):, East-West Rates(mm):"
        )

    use_cols = [lon_col, lat_col, ve_col, vn_col]
    if station_col:
        use_cols.append(station_col)
    if fault_col:
        use_cols.append(fault_col)

    plot_df = df[use_cols].copy().dropna(subset=[lon_col, lat_col, ve_col, vn_col])

    rename_map = {lon_col: "lon", lat_col: "lat", ve_col: "ve_mm_yr", vn_col: "vn_mm_yr"}
    if station_col:
        rename_map[station_col] = "station"
    if fault_col:
        rename_map[fault_col] = "fault"
    plot_df = plot_df.rename(columns=rename_map)

    # Exclude stations CSTA and TSFT
    if "station" in plot_df.columns:
        plot_df["station"] = plot_df["station"].astype(str).str.strip()
        plot_df = plot_df[~plot_df["station"].str.upper().isin(EXCLUDED_STATIONS)]

    for c in ["lon", "lat", "ve_mm_yr", "vn_mm_yr"]:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df = plot_df.dropna(subset=["lon", "lat", "ve_mm_yr", "vn_mm_yr"])

    if plot_df.empty:
        raise SystemExit("No valid rows found after cleaning/exclusions.")

    # Project to Web Mercator (meters)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_m, y_m = transformer.transform(plot_df["lon"].values, plot_df["lat"].values)

    # Velocity vectors (m/yr)
    ve_m_yr = plot_df["ve_mm_yr"].values / 1000.0
    vn_m_yr = plot_df["vn_mm_yr"].values / 1000.0
    speed_mm_yr = np.sqrt(plot_df["ve_mm_yr"].values**2 + plot_df["vn_mm_yr"].values**2)
    plot_df["speed_mm_yr"] = speed_mm_yr

    # Dynamic arrow scaling
    vec_mag = np.sqrt(ve_m_yr**2 + vn_m_yr**2)
    data_x_span = max(np.max(x_m) - np.min(x_m), 1.0)
    data_y_span = max(np.max(y_m) - np.min(y_m), 1.0)

    if USE_FIXED_VIEW:
        map_span_for_scale = max(VIEW_WIDTH_KM * 1000.0, VIEW_HEIGHT_KM * 1000.0)
    else:
        map_span_for_scale = max(data_x_span, data_y_span)

    median_mag = np.median(vec_mag[vec_mag > 0]) if np.any(vec_mag > 0) else 1e-12
    display_years = (TARGET_ARROW_FRACTION_OF_MAP * map_span_for_scale) / median_mag
    u_plot = ve_m_yr * display_years
    v_plot = vn_m_yr * display_years

    # Shared colormap/norm so station dots match arrow colors
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(np.min(speed_mm_yr)), vmax=float(np.max(speed_mm_yr)))

    fig, ax = plt.subplots(figsize=(12, 9))

    # Points by fault class (color = speed, marker = class)
    if "fault" in plot_df.columns:
        fault_vals = plot_df["fault"].astype(str).str.strip().str.lower()
        fw = plot_df[fault_vals == "footwall"]
        hw = plot_df[fault_vals != "footwall"]  # treat all non-footwall as Hangingwall

        if not fw.empty:
            xf, yf = transformer.transform(fw["lon"].values, fw["lat"].values)
            ax.scatter(
                xf, yf,
                c=fw["speed_mm_yr"].values, cmap=cmap, norm=norm,
                s=46, marker="o", edgecolor="black", linewidth=0.7,
                alpha=0.95, zorder=6
            )
        if not hw.empty:
            xh, yh = transformer.transform(hw["lon"].values, hw["lat"].values)
            ax.scatter(
                xh, yh,
                c=hw["speed_mm_yr"].values, cmap=cmap, norm=norm,
                s=46, marker="s", edgecolor="black", linewidth=0.7,
                alpha=0.95, zorder=6
            )
    else:
        ax.scatter(
            x_m, y_m,
            c=speed_mm_yr, cmap=cmap, norm=norm,
            s=46, edgecolor="black", linewidth=0.7, alpha=0.95, zorder=6
        )

    q = ax.quiver(
        x_m, y_m, u_plot, v_plot, speed_mm_yr,
        angles="xy", scale_units="xy", scale=1,
        cmap=cmap, norm=norm, width=0.0030,
        headwidth=5.0, headlength=7.0, headaxislength=6.0,
        pivot="tail", zorder=7
    )

    # More readable colorbar (slightly smaller)
    cax = inset_axes(ax, width="4.6%", height="50%", loc="lower right", borderpad=1.0)
    cbar = plt.colorbar(q, cax=cax)
    cbar.set_label("Horizontal Velocity (mm/yr)", fontsize=13, fontweight="bold", labelpad=12)
    cbar.ax.tick_params(labelsize=12, width=1.4, length=5, colors="black")
    cbar.outline.set_linewidth(1.4)
    cax.set_facecolor((1, 1, 1, 1.0))
    for spine in cax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2))

    # Plot Long Point Fault from KML
    fault_lines = parse_kml_fault_lines(KML_PATH)
    if fault_lines:
        for i, ll in enumerate(fault_lines):
            fx, fy = transformer.transform(ll[:, 0], ll[:, 1])
            ax.plot(fx, fy, "r-", lw=2, zorder=8, label="Long Point Fault" if i == 0 else None)
    else:
        print("No LineString geometries found in KML; fault line not plotted.")

    # Labels: larger + white halo for readability + station-specific offsets
    if SHOW_LABELS and "station" in plot_df.columns:
        x0, x1 = np.min(x_m), np.max(x_m)
        y0, y1 = np.min(y_m), np.max(y_m)
        dx = x1 - x0
        dy = y1 - y0
        ox = 0.006 * dx
        oy = 0.006 * dy

        for _, r in plot_df.iterrows():
            xi, yi = transformer.transform(r["lon"], r["lat"])
            st = str(r["station"]).strip()
            ex, ey = LABEL_OFFSETS_M.get(st.upper(), (0.0, 0.0))
            txt = ax.text(
                xi + ox + ex, yi + oy + ey, st,
                fontsize=10, fontweight="bold", color="black",
                ha="left", va="bottom", zorder=10
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

    # Extents
    if USE_FIXED_VIEW:
        cx = float(np.median(x_m))
        cy = float(np.median(y_m))
        half_w = (VIEW_WIDTH_KM * 1000.0) / 2.0
        half_h = (VIEW_HEIGHT_KM * 1000.0) / 2.0
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)
    else:
        pad_x = data_x_span * 0.15
        pad_y = data_y_span * 0.15
        ax.set_xlim(np.min(x_m) - pad_x, np.max(x_m) + pad_x)
        ax.set_ylim(np.min(y_m) - pad_y, np.max(y_m) + pad_y)

    # Basemap
    if USE_BASEMAP and HAS_CONTEXTILY:
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            crs="EPSG:3857",
            zoom=BASEMAP_ZOOM,
            interpolation="bilinear"
        )
    elif USE_BASEMAP and not HAS_CONTEXTILY:
        print("contextily not installed; plotting without basemap.")

    # Scale bar and north arrow
    add_distance_scale(ax, length_km=SCALE_BAR_KM)
    add_north_arrow(ax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("GPS Velocity Vector Map", pad=10, fontsize=15, fontweight="bold")
    apply_lonlat_axes(ax)

    # Legend with class markers (shape meaning)
    if "fault" in plot_df.columns or fault_lines:
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=8, linestyle='None', label='Footwall'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=8, linestyle='None', label='Hangingwall'),
            Line2D([0], [0], color='red', lw=2, label='Long Point Fault')
        ]
        leg = ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.9)
        leg.get_frame().set_facecolor("white")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()