import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
folder_path = r"C:\Users\jt00784\Desktop\Well Data\*.csv"
OUTPUT_DIR = r"C:\Users\jt00784\Desktop\Well Data\Output"
WINDOW_SIZE = 30
YEAR_THRESHOLD = 2018

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = glob.glob(folder_path)
all_plots_data = []

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
        df_2018 = df[df["Date"] >= f"{YEAR_THRESHOLD}-01-01"]
        if len(df_2018) > 2:
            trend_2018 = calculate_regression(df_2018, df_2018["Year"].values, "Depth_ft")
        else:
            trend_2018 = None

        # Store data for final figure
        plots = [
            (df["Date"], y, full_trend, f"{name} - Full Record", "Full"),
            (df_smooth["Date"], df_smooth["Smooth_m"].values, smooth_trend, f"{name} - Smoothed (30-day)", "Smoothed"),
        ]
        
        if trend_2018:
            plots.append((df_2018["Date"], df_2018["Depth_m"].values, trend_2018, f"{name} - {YEAR_THRESHOLD}+", f"{YEAR_THRESHOLD}+"))
        
        all_plots_data.append((name, plots))
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
    t_val = stats.t.ppf(0.975, df=n-2)
    residuals = y_ft - (slope_ft * x + intercept_ft)
    s_err = np.sqrt(np.sum(residuals**2) / (n-2))
    x_mean = np.mean(x)
    conf_m = t_val * s_err * 0.3048 * np.sqrt(1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2))
    
    lower = y_pred_m - conf_m
    upper = y_pred_m + conf_m
    
    slope_mm_per_year = slope_ft * 304.8
    ci_mm = t_val * std_err_ft * 304.8
    
    return {
        'pred': y_pred_m,
        'lower': lower,
        'upper': upper,
        'slope_mm_yr': slope_mm_per_year,
        'ci_mm': ci_mm,
        'r_squared': r_value**2,
        'p_value': p_value,
        'data_points': n,
        'x': x
    }

def create_individual_plots(name, plots):
    fig, axes = plt.subplots(1, len(plots), figsize=(7*len(plots), 6))
    if len(plots) == 1:
        axes = [axes]
    
    for ax, (dates, depths, trend, title, label) in zip(axes, plots):
        ax.plot(dates, depths, 'o-', alpha=0.5, markersize=3, label="Observed")
        
        if trend:
            ax.plot(dates, trend['pred'], 'r-', linewidth=2, label=f"Trend")
            ax.fill_between(dates, trend['lower'], trend['upper'], color="red", alpha=0.2, label="95% CI")
            
            # Add text box with statistics
            textstr = f"Slope: {trend['slope_mm_yr']:.2f} ± {trend['ci_mm']:.2f} mm/yr\n"
            textstr += f"R²: {trend['r_squared']:.3f}\n"
            textstr += f"p-value: {trend['p_value']:.4f}\n"
            textstr += f"n = {trend['data_points']}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
        
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Depth (m)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{name}_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def generate_summary_statistics(all_plots_data):
    summary_data = []
    
    for well_name, plots in all_plots_data:
        for dates, depths, trend, title, label in plots:
            if trend:
                summary_data.append({
                    'Well_Name': well_name,
                    'Analysis_Type': label,
                    'Slope_mm_per_year': trend['slope_mm_yr'],
                    'Slope_CI_mm': trend['ci_mm'],
                    'R_squared': trend['r_squared'],
                    'P_value': trend['p_value'],
                    'Data_Points': trend['data_points'],
                    'Significant': 'Yes' if trend['p_value'] < 0.05 else 'No'
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

# Create combined figure
if all_plots_data:
    print(f"\nCreating combined figure for {len(all_plots_data)} wells...")
    
    # Generate summary statistics CSV
    summary_df = generate_summary_statistics(all_plots_data)
    print("Summary Statistics:")
    print(summary_df.to_string(index=False))

if all_plots_data:
    print(f"\nCreating combined figure for {len(all_plots_data)} wells...")
    
    n_wells = len(all_plots_data)
    n_plots_per_well = max(len(plots) for _, plots in all_plots_data)
    
    fig, axes = plt.subplots(n_wells, n_plots_per_well, figsize=(7*n_plots_per_well, 6*n_wells))
    if n_wells == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_plots_per_well == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i, (well_name, plots) in enumerate(all_plots_data):
        for j, (dates, depths, trend, title, label) in enumerate(plots):
            ax = axes[i, j]
            ax.plot(dates, depths, 'o-', alpha=0.5, markersize=2, label="Observed")
            
            if trend:
                ax.plot(dates, trend['pred'], 'r-', linewidth=2, 
                       label=f"Trend: {trend['slope_mm_yr']:.1f} ± {trend['ci_mm']:.1f} mm/yr")
                ax.fill_between(dates, trend['lower'], trend['upper'], color="red", alpha=0.2, label="95% CI")
            
            ax.invert_yaxis()
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Depth (m)", fontsize=9)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR, "combined_trends.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {combined_path}\n")
    plt.show()
else:
    print("No valid files found to process.")

