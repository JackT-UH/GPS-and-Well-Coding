import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
folder_path = r"C:\Users\jt00784\Desktop\Well Data\*.csv"
OUTPUT_DIR = r"C:\Users\jt00784\Desktop\Well Data\Output"
COMBINED_DIR = r"C:\Users\jt00784\Desktop\Well Data\Output\Combined"
WINDOW_SIZE = 30
YEAR_THRESHOLD = 2018

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

files = glob.glob(folder_path)
all_plots_data = []
piezometer_data = []
extensometer_data = []

def process_file(file_path):
    try:
        name = os.path.basename(file_path).replace(".csv", "")
        df = pd.read_csv(file_path)
        df.columns = ["Date", "Depth_ft"]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna()
        df["Depth_m"] = df["Depth_ft"] * 0.3048
        df["Year"] = df["Date"].dt.year + df["Date"].dt.dayofyear / 365.25

        x = df["Year"].values
        y = df["Depth_m"].values

        # Full record regression
        full_trend = calculate_regression(df, x, "Depth_ft")

        # 30-day rolling smoothed regression
        df["Smooth_m"] = df["Depth_m"].rolling(window=WINDOW_SIZE, center=True).mean()
        df_smooth = df.dropna(subset=["Smooth_m"])
        smooth_trend = calculate_regression(df_smooth, df_smooth["Year"].values, "Smooth_m", is_meters=True)

        # 2018+ regression
        df_2018 = df[df["Date"] >= f"{YEAR_THRESHOLD}-01-01"].copy()
        if len(df_2018) > 2:
            trend_2018 = calculate_regression(df_2018, df_2018["Year"].values, "Depth_ft")
        else:
            trend_2018 = None

        # 2018+ smoothed regression
        df_2018["Smooth_m"] = df_2018["Depth_m"].rolling(window=WINDOW_SIZE, center=True).mean()
        df_2018_smooth = df_2018.dropna(subset=["Smooth_m"])
        if len(df_2018_smooth) > 2:
            trend_2018_smooth = calculate_regression(df_2018_smooth, df_2018_smooth["Year"].values, "Smooth_m", is_meters=True)
        else:
            trend_2018_smooth = None

        # Store data for final figure
        plots = [
            (df["Date"], y, full_trend, f"{name} - Full Record", "Full"),
            (df_smooth["Date"], df_smooth["Smooth_m"].values, smooth_trend, f"{name} - Smoothed (30-day)", "Smoothed"),
        ]

        if trend_2018:
            plots.append((df_2018["Date"], df_2018["Depth_m"].values, trend_2018, f"{name} - {YEAR_THRESHOLD}+", f"{YEAR_THRESHOLD}+"))

        if trend_2018_smooth:
            plots.append((df_2018_smooth["Date"], df_2018_smooth["Smooth_m"].values, trend_2018_smooth, f"{name} - {YEAR_THRESHOLD}+ Smoothed", f"{YEAR_THRESHOLD}+_Smoothed"))

        all_plots_data.append((name, plots))

        # Categorize by type (piezometer or extensometer)
        if "piezometer" in name.lower():
            piezometer_data.append((name, plots))
        elif "extensometer" in name.lower():
            extensometer_data.append((name, plots))

        print(f"✓ Processed: {name}")
        return True

    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        return False

def calculate_regression(df, x, depth_column, is_meters=False):
    if len(x) < 3:
        return None

    # Convert to feet for regression if needed
    y_ft = df[depth_column].values if "ft" in depth_column else df[depth_column].values / 0.3048

    slope_ft, intercept_ft, r_value, p_value, std_err_ft = stats.linregress(x, y_ft)

    # Predictions in meters
    y_pred_m = (intercept_ft + slope_ft * x) * 0.3048

    # Confidence intervals
    n = len(x)
    t_val = stats.t.ppf(0.975, df=n - 2)
    residuals = y_ft - (slope_ft * x + intercept_ft)
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    x_mean = np.mean(x)
    conf_m = t_val * s_err * 0.3048 * np.sqrt(1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2))

    lower = y_pred_m - conf_m
    upper = y_pred_m + conf_m

    slope_mm_per_year = slope_ft * 304.8
    ci_mm = t_val * std_err_ft * 304.8

    return {
        "pred": y_pred_m,
        "lower": lower,
        "upper": upper,
        "slope_mm_yr": slope_mm_per_year,
        "ci_mm": ci_mm,
        "r_squared": r_value**2,
        "p_value": p_value,
        "data_points": n,
        "x": x
    }

def create_individual_plots(name, plots):
    fig, axes = plt.subplots(1, len(plots), figsize=(7 * len(plots), 6))
    if len(plots) == 1:
        axes = [axes]

    for ax, (dates, depths, trend, title, label) in zip(axes, plots):
        ax.plot(dates, depths, "o-", alpha=0.5, markersize=3, label="Observed")

        if trend:
            ax.plot(dates, trend["pred"], "r-", linewidth=2, label="Trend")
            ax.fill_between(dates, trend["lower"], trend["upper"], color="red", alpha=0.2, label="95% CI")

            textstr = f"Slope: {trend['slope_mm_yr']:.2f} ± {trend['ci_mm']:.2f} mm/yr\n"
            textstr += f"R²: {trend['r_squared']:.3f}\n"
            textstr += f"p-value: {trend['p_value']:.4f}\n"
            textstr += f"n = {trend['data_points']}"
            ax.text(
                0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), fontsize=9
            )

        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Depth (m)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{name}_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

def create_combined_type_plots(all_type_data):
    """Create two combined images including BOTH piezometer and extensometer rows."""
    if not all_type_data:
        return

    # Image 1: Full Record + Smoothed
    fig, axes = plt.subplots(len(all_type_data), 2, figsize=(14, 5 * len(all_type_data)))
    if len(all_type_data) == 1:
        axes = axes.reshape(1, -1)

    for i, (well_name, plots) in enumerate(all_type_data):
        well_type = "Piezometer" if "piezometer" in well_name.lower() else ("Extensometer" if "extensometer" in well_name.lower() else "Well")

        # Full Record
        dates, depths, trend, _, _ = plots[0]
        ax = axes[i, 0]
        ax.plot(dates, depths, "o-", alpha=0.5, markersize=3, label="Observed")
        if trend:
            ax.plot(dates, trend["pred"], "r-", linewidth=2, label="Trend")
            ax.fill_between(dates, trend["lower"], trend["upper"], color="red", alpha=0.2, label="95% CI")
        ax.invert_yaxis()
        ax.set_title(f"{well_name} [{well_type}] - Full Record", fontsize=10, fontweight="bold")
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        # Smoothed
        dates, depths, trend, _, _ = plots[1]
        ax = axes[i, 1]
        ax.plot(dates, depths, "o-", alpha=0.5, markersize=3, label="Observed")
        if trend:
            ax.plot(dates, trend["pred"], "r-", linewidth=2, label="Trend")
            ax.fill_between(dates, trend["lower"], trend["upper"], color="red", alpha=0.2, label="95% CI")
        ax.invert_yaxis()
        ax.set_title(f"{well_name} [{well_type}] - Smoothed (30-day)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = os.path.join(COMBINED_DIR, "CombinedTypes_FullRecord_and_Smoothed.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

    # Image 2: 2018+ + 2018+ Smoothed
    fig, axes = plt.subplots(len(all_type_data), 2, figsize=(14, 5 * len(all_type_data)))
    if len(all_type_data) == 1:
        axes = axes.reshape(1, -1)

    for i, (well_name, plots) in enumerate(all_type_data):
        well_type = "Piezometer" if "piezometer" in well_name.lower() else ("Extensometer" if "extensometer" in well_name.lower() else "Well")

        # 2018+
        ax = axes[i, 0]
        if len(plots) > 2:
            dates, depths, trend, _, _ = plots[2]
            ax.plot(dates, depths, "o-", alpha=0.5, markersize=3, label="Observed")
            if trend:
                ax.plot(dates, trend["pred"], "r-", linewidth=2, label="Trend")
                ax.fill_between(dates, trend["lower"], trend["upper"], color="red", alpha=0.2, label="95% CI")
            ax.invert_yaxis()
            ax.set_title(f"{well_name} [{well_type}] - {YEAR_THRESHOLD}+", fontsize=10, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
        else:
            ax.set_title(f"{well_name} [{well_type}] - {YEAR_THRESHOLD}+ (No data)", fontsize=10, fontweight="bold")
            ax.set_axis_off()

        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        # 2018+ Smoothed
        ax = axes[i, 1]
        if len(plots) > 3:
            dates, depths, trend, _, _ = plots[3]
            ax.plot(dates, depths, "o-", alpha=0.5, markersize=3, label="Observed")
            if trend:
                ax.plot(dates, trend["pred"], "r-", linewidth=2, label="Trend")
                ax.fill_between(dates, trend["lower"], trend["upper"], color="red", alpha=0.2, label="95% CI")
            ax.invert_yaxis()
            ax.set_title(f"{well_name} [{well_type}] - {YEAR_THRESHOLD}+ Smoothed", fontsize=10, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
        else:
            ax.set_title(f"{well_name} [{well_type}] - {YEAR_THRESHOLD}+ Smoothed (No data)", fontsize=10, fontweight="bold")
            ax.set_axis_off()

        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = os.path.join(COMBINED_DIR, "CombinedTypes_2018Plus_and_Smoothed.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

def generate_summary_statistics(all_plots_data):
    summary_data = []

    for well_name, plots in all_plots_data:
        for dates, depths, trend, title, label in plots:
            if trend:
                summary_data.append({
                    "Well_Name": well_name,
                    "Analysis_Type": label,
                    "Slope_mm_per_year": trend["slope_mm_yr"],
                    "Slope_CI_mm": trend["ci_mm"],
                    "R_squared": trend["r_squared"],
                    "P_value": trend["p_value"],
                    "Data_Points": trend["data_points"],
                    "Significant": "Yes" if trend["p_value"] < 0.05 else "No"
                })

    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(OUTPUT_DIR, "well_statistics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}\n")
    return summary_df

# Process all files
print(f"Processing {len(files)} files...\n")
for file in sorted(files):
    if process_file(file):
        name, plots = all_plots_data[-1]
        create_individual_plots(name, plots)

# Generate summary statistics CSV
if all_plots_data:
    print("\nGenerating summary statistics...")
    summary_df = generate_summary_statistics(all_plots_data)
    print("Summary Statistics:")
    print(summary_df.to_string(index=False))

# Create combined type plots (piezometer + extensometer in SAME images)
print("\nCreating combined type-specific plots...")
combined_type_data = piezometer_data + extensometer_data
if combined_type_data:
    create_combined_type_plots(combined_type_data)

# Create overall combined figure
if all_plots_data:
    print(f"\nCreating overall combined figure for {len(all_plots_data)} wells...")

    # Force order: piezometer first, then extensometer, then any others
    piezo_names = {name for name, _ in piezometer_data}
    exten_names = {name for name, _ in extensometer_data}
    other_data = [(name, plots) for name, plots in all_plots_data if name not in piezo_names and name not in exten_names]
    ordered_plots_data = piezometer_data + extensometer_data + other_data

    n_wells = len(ordered_plots_data)
    n_plots_per_well = max(len(plots) for _, plots in ordered_plots_data)

    fig, axes = plt.subplots(n_wells, n_plots_per_well, figsize=(7 * n_plots_per_well, 6 * n_wells))
    if n_wells == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_plots_per_well == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, (well_name, plots) in enumerate(ordered_plots_data):
        for j, (dates, depths, trend, title, label) in enumerate(plots):
            ax = axes[i, j]
            ax.plot(dates, depths, "o-", alpha=0.5, markersize=2, label="Observed")

            if trend:
                ax.plot(
                    dates, trend["pred"], "r-", linewidth=2,
                    label=f"Trend: {trend['slope_mm_yr']:.1f} ± {trend['ci_mm']:.1f} mm/yr"
                )
                ax.fill_between(dates, trend["lower"], trend["upper"], color="red", alpha=0.2, label="95% CI")

            ax.invert_yaxis()
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Depth (m)", fontsize=9)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR, "combined_trends.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {combined_path}\n")

print("✓ Processing complete!")