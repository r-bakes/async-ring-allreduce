import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os
import glob

def format_bytes(x, pos):
    """Formats axis ticks into readable sizes (KB, MB, GB)."""
    if x == 0:
        return '0 B'
    
    # 1 KB = 1024 Bytes logic
    if x >= 1024**3:
        return f'{x / 1024**3:.0f} GB'
    elif x >= 1024**2:
        return f'{x / 1024**2:.0f} MB'
    elif x >= 1024:
        return f'{x / 1024:.0f} KB'
    else:
        return f'{x:.0f} B'

def save_single_plot(data, output_path, title, hue_col=None):
    """Helper to generate and save a single S-curve plot."""
    plt.figure(figsize=(10, 6))
    
    p = sns.lineplot(
        data=data,
        x='input_bytes',
        y='Bandwidth (GB/s)',
        hue=hue_col,
        markers=True,
        dashes=True,
        linewidth=2.5,
        markersize=8,
        palette="hls"  # Good default palette for distinctions
    )

    # Beautify Axes
    p.set_xscale('log')
    p.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    
    plt.grid(True, which="minor", ls="--", alpha=0.3)
    plt.grid(True, which="major", ls="-", alpha=0.8)

    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Message Size", fontsize=12, labelpad=10)
    plt.ylabel("Effective Bandwidth (GB/s)", fontsize=12, labelpad=10)
    
    # Place legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue_col.capitalize() if hue_col else None)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close() # Important: close figure to free memory
    print(f"Saved plot: {output_path}")

def plot_scurve(input_dir, output_file_base):
    # 1. Setup Theme
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 2. Load Data
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Use filename as a label (e.g., 'result_4gpu.csv' -> 'result_4gpu')
            source_name = os.path.splitext(os.path.basename(filename))[0]
            df['Configuration'] = source_name
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not df_list:
        return

    data = pd.concat(df_list, ignore_index=True)

    # 3. Data Preprocessing
    data['Bandwidth (GB/s)'] = data['throughput'] / 1000.0
    
    # Prepare base filename prefix (strip extension if present)
    base_name = os.path.splitext(output_file_base)[0]

    # 4. PLOT TYPE A: One plot per CSV Configuration (Comparing algorithms within a file)
    unique_configs = data['Configuration'].unique()
    print(f"\n--- Generating Configuration Comparison Plots ({len(unique_configs)}) ---")
    for config in unique_configs:
        subset = data[data['Configuration'] == config]
        out_name = f"{base_name}_conf_{config}.png"
        save_single_plot(subset, out_name, f"Bandwidth S-Curve: {config}", hue_col='impl')

    # 5. PLOT TYPE B: One plot per Implementation (Comparing configurations/scaling)
    unique_impls = data['impl'].unique()
    print(f"\n--- Generating Implementation Scaling Plots ({len(unique_impls)}) ---")
    for impl in unique_impls:
        subset = data[data['impl'] == impl]
        out_name = f"{base_name}_impl_{impl}.png"
        save_single_plot(subset, out_name, f"Bandwidth S-Curve: {impl}", hue_col='Configuration')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot S-Curve from benchmark CSVs")
    parser.add_argument("input_dir", type=str, help="Directory containing CSV results")
    parser.add_argument("--output", type=str, default="s_curve.png", help="Base output filename (will be suffixed)")
    
    args = parser.parse_args()
    plot_scurve(args.input_dir, args.output)