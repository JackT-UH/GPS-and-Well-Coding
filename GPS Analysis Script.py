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
folder_path = r"C:\Users\jt00784\Desktop\GPS Data\LPF Data\*.col"
OUTPUT_DIR = r"C:\Users\jt00784\Desktop\GPS Data\Output"
DIRECTIONAL_DIR = r"C:\Users\jt00784\Desktop\GPS Data\Output\Directional_Components"
DESEASONALIZATION_DIR = r"C:\Users\jt00784\Desktop\GPS Data\Output\Deseasonalization"
WINDOW_SIZE = 30
YEAR_THRESHOLD = 2018

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DIRECTIONAL_DIR, exist_ok=True)
os.makedirs(DESEASONALIZATION_DIR, exist_ok=True)

files = glob.glob(folder_path)
all_plots_data = []
comparison_data = []

def deseasonalize_data(df, column, window=365):
    """Deseasonalize data by removing seasonal component."""
    # Calculate seasonal component
    df[f'{column}_seasonal'] = df[column].rolling(window=window, center=True).mean()
    
    # Deseasonalized = Original - Seasonal
    df[f'{column}_deseas'] = df[column] - df[f'{column}_seasonal']
    
    return df

def process_file(file_path):
    try:
        name = os.path.basename(file_path).replace(".col", "")
        df = pd.read_csv(file_path, sep=r'\s+', header=0)
        
        # Rename columns for clarity
        df.columns = ["Decimal_Year", "NS_cm", "EW_cm", "UD_cm", "sigma_NS", "sigma_EW", "sigma_UD"]
        
        # Convert to datetime from decimal year
        df["Date"] = pd.to_datetime(df["Decimal_Year"].apply(lambda x: f"{int(x)}-01-01"), errors="coerce") + \
                     pd.to_timedelta((df["Decimal_Year"] - df["Decimal_Year"].astype(int)) * 365.25, unit='D')
        
        df = df.dropna(subset=["NS_cm", "EW_cm", "UD_cm"])
        df["Year"] = df["Decimal_Year"].values
        
        directional_plots = []
        deseasonalization_plots = []
        
        # Analyze each direction: UD, NS, EW
        for direction, column, sigma_column in [("UD (Vertical)", "UD_cm", "sigma_UD"), 
                                                 ("NS (North-South)", "NS_cm", "sigma_NS"), 
                                                 ("EW (East-West)", "EW_cm", "sigma_EW")]:
            # Data stays in cm for plotting
            y_values = df[column].values
            x = df["Year"].values
            weights = df[sigma_column].values  # Use measurement uncertainties (in cm)
            
            # ===== DIRECTIONAL COMPONENTS =====
            
            # Full record analysis - Weighted Least Squares (using cm data, slope in mm/yr)
            trend_full = calculate_regression(df, column, x, weights, data_in_cm=True, use_weighted=True)
            directional_plots.append((df["Date"], y_values, trend_full, f"{name} - {direction} (Full Record)", f"{direction}_Full"))
            
            # Smoothed analysis
            smooth_col = f"{column}_smooth"
            df[smooth_col] = df[column].rolling(window=WINDOW_SIZE, center=True).mean()  # Smooth cm data
            df_smooth = df.dropna(subset=[smooth_col])
            weights_smooth = df_smooth[sigma_column].values
            trend_smooth = calculate_regression(df_smooth, smooth_col, df_smooth["Year"].values, weights_smooth, data_in_cm=True, use_weighted=True)
            directional_plots.append((df_smooth["Date"], df_smooth[smooth_col].values, trend_smooth, f"{name} - {direction} (Smoothed)", f"{direction}_Smooth"))
            
            # 2018+ analysis
            df_2018 = df[df["Decimal_Year"] >= YEAR_THRESHOLD]
            if len(df_2018) > 2:
                x_2018 = df_2018["Year"].values
                weights_2018 = df_2018[sigma_column].values
                trend_2018 = calculate_regression(df_2018, column, x_2018, weights_2018, data_in_cm=True, use_weighted=True)
                directional_plots.append((df_2018["Date"], df_2018[column].values, trend_2018, f"{name} - {direction} ({YEAR_THRESHOLD}+)", f"{direction}_2018+"))
                
                # 2018+ Smoothed analysis
                df_2018[f'{column}_smooth_2018'] = df_2018[column].rolling(window=WINDOW_SIZE, center=True).mean()
                df_2018_smooth = df_2018.dropna(subset=[f'{column}_smooth_2018'])
                if len(df_2018_smooth) > 2:
                    weights_2018_smooth = df_2018_smooth[sigma_column].values
                    trend_2018_smooth = calculate_regression(df_2018_smooth, f'{column}_smooth_2018', df_2018_smooth["Year"].values, weights_2018_smooth, data_in_cm=True, use_weighted=True)
                    directional_plots.append((df_2018_smooth["Date"], df_2018_smooth[f'{column}_smooth_2018'].values, trend_2018_smooth, f"{name} - {direction} ({YEAR_THRESHOLD}+ Smoothed)", f"{direction}_2018+_Smooth"))
            
            # ===== DESEASONALIZATION ANALYSIS =====
            
            # Deseasonalize the data
            df = deseasonalize_data(df, column, window=365)
            deseas_col = f"{column}_deseas"
            trend_deseas = calculate_regression(df, deseas_col, x, weights, data_in_cm=True, use_weighted=True)
            # Remove statistics from deseasonalized plots (pass trend as None for display)
            deseasonalization_plots.append((df["Date"], df[deseas_col].values, None, f"{name} - {direction} (Seasonal variation)", f"{direction}_Deseas"))
        
        all_plots_data.append((name, directional_plots, deseasonalization_plots))
        print(f"✓ Processed: {name}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        return False

def calculate_regression(df, column, x, weights, data_in_cm=False, use_weighted=False):
    """
    Calculate regression with rates in mm/yr.
    Handles NaN values by filtering them out before regression.
    """
    # Remove NaN values first
    valid_mask = ~np.isnan(df[column].values)
    
    if np.sum(valid_mask) < 3:
        return None
    
    y_values = df[column].values[valid_mask]
    x_clean = x[valid_mask]
    weights_clean = weights[valid_mask]
    
    # If data is in cm, convert to mm for calculations
    if data_in_cm:
        y_values = y_values * 10  # Convert cm to mm
    
    n = len(x_clean)
    
    if use_weighted and weights_clean is not None:
        # Weighted Least Squares
        # Use uncertainties in cm, convert to weights (inverse of variance)
        w_values = weights_clean  # Keep in cm for weighting
        w = 1.0 / (w_values ** 2)
        w = w / np.sum(w) * len(w)  # Normalize
        
        # Weighted regression using polyfit
        coeffs = np.polyfit(x_clean, y_values, 1, w=w)
        slope = coeffs[0]  # mm/yr
        intercept = coeffs[1]
        
        y_pred = intercept + slope * x_clean
        
        # Weighted residuals
        residuals = y_values - y_pred
        s_err = np.sqrt(np.sum(w * residuals**2) / (n-2))
        
        t_val = stats.t.ppf(0.975, df=n-2)
        x_mean = np.average(x_clean, weights=w)
        x_var = np.average((x_clean - x_mean)**2, weights=w)
        
        # Weighted confidence intervals
        conf = t_val * s_err * np.sqrt(1/np.sum(w) + (x_clean - x_mean)**2 / (np.sum(w) * x_var))
        lower = y_pred - conf
        upper = y_pred + conf
        
        # Calculate R-squared for weighted regression
        ss_res = np.sum(w * residuals**2)
        y_weighted_mean = np.average(y_values, weights=w)
        ss_tot = np.sum(w * (y_values - y_weighted_mean)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        se_slope = s_err / np.sqrt(np.sum(w) * x_var)
        ci_slope = t_val * se_slope
        
        p_value = 2 * (1 - stats.t.cdf(abs(slope / se_slope), df=n-2))
        
    else:
        # Standard Linear Regression (fallback)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_values)
        y_pred = intercept + slope * x_clean
        
        # Confidence intervals
        t_val = stats.t.ppf(0.975, df=n-2)
        residuals = y_values - (slope * x_clean + intercept)
        s_err = np.sqrt(np.sum(residuals**2) / (n-2))
        x_mean = np.mean(x_clean)
        conf = t_val * s_err * np.sqrt(1/n + (x_clean - x_mean)**2 / np.sum((x_clean - x_mean)**2))
        
        lower = y_pred - conf
        upper = y_pred + conf
        
        r_squared = r_value**2
        se_slope = std_err
        ci_slope = t_val * se_slope
    
    return {
        'pred': y_pred,
        'lower': lower,
        'upper': upper,
        'slope_mm_yr': slope,  # Slope in mm/yr
        'ci_mm': ci_slope,
        'r_squared': r_squared,
        'p_value': p_value,
        'data_points': n,
        'x': x_clean
    }

def create_individual_plots(name, plots, save_to_dir, title_prefix=""):
    """Create individual plots for each analysis type."""
    n_plots = len(plots)
    n_rows = (n_plots + 2) // 3  # 3 columns
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (dates, values, trend, title, label) in enumerate(plots):
        ax = axes[i]
        ax.plot(dates, values, 'o-', alpha=0.5, markersize=3, label="Observed")
        
        if trend:
            # Convert trend predictions from mm back to cm for plotting
            y_pred_cm = trend['pred'] / 10
            lower_cm = trend['lower'] / 10
            upper_cm = trend['upper'] / 10
            
            # Get valid dates corresponding to cleaned x values
            valid_dates = dates[~np.isnan(values[:len(dates)])]
            if len(y_pred_cm) <= len(valid_dates):
                plot_dates = valid_dates[:len(y_pred_cm)]
            else:
                plot_dates = valid_dates
            
            ax.plot(plot_dates, y_pred_cm[:len(plot_dates)], 'b-', linewidth=2, label=f"WLS Trend")
            ax.fill_between(plot_dates, lower_cm[:len(plot_dates)], upper_cm[:len(plot_dates)], 
                           color="blue", alpha=0.2, label="95% CI")
            
            # Add text box with statistics (slope in mm/yr)
            textstr = f"Slope: {trend['slope_mm_yr']:.2f} ± {trend['ci_mm']:.2f} mm/yr\n"
            textstr += f"R²: {trend['r_squared']:.3f}\n"
            textstr += f"p-value: {trend['p_value']:.4f}\n"
            textstr += f"n = {trend['data_points']}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=8)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Displacement (cm)", fontsize=8)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(save_to_dir, f"{name}_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def generate_summary_statistics(all_plots_data):
    """Generate summary statistics for all analyses."""
    summary_data = []
    
    for station_name, directional_plots, deseasonalization_plots in all_plots_data:
        # Add directional component statistics
        for dates, values, trend, title, label in directional_plots:
            if trend:
                summary_data.append({
                    'Station_Name': station_name,
                    'Component': 'Directional',
                    'Analysis_Type': label,
                    'Slope_mm_per_year': trend['slope_mm_yr'],
                    'Slope_CI_mm': trend['ci_mm'],
                    'R_squared': trend['r_squared'],
                    'P_value': trend['p_value'],
                    'Data_Points': trend['data_points'],
                    'Significant': 'Yes' if trend['p_value'] < 0.05 else 'No'
                })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(OUTPUT_DIR, "gps_statistics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}\n")
    return summary_df

# Process all files
print(f"Processing {len(files)} files...\n")
for file in sorted(files):
    if process_file(file):
        station_name, directional_plots, deseasonalization_plots = all_plots_data[-1]
        
        # Save directional components
        if directional_plots:
            create_individual_plots(station_name, directional_plots, DIRECTIONAL_DIR, "Directional Components")
        
        # Save deseasonalization analysis
        if deseasonalization_plots:
            create_individual_plots(station_name, deseasonalization_plots, DESEASONALIZATION_DIR, "Deseasonalization Analysis")

# Create combined figure
if all_plots_data:
    print(f"\nCreating final summary...")
    
    # Generate summary statistics CSV
    summary_df = generate_summary_statistics(all_plots_data)
    print("Summary Statistics:")
    print(summary_df.to_string(index=False))
    print(f"\n✓ Processing complete!")
    print(f"✓ Directional components saved to: {DIRECTIONAL_DIR}")
    print(f"✓ Deseasonalization analysis saved to: {DESEASONALIZATION_DIR}")
    print(f"✓ Summary statistics saved to: {OUTPUT_DIR}")
else:
    print("No valid files found to process.")