import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os


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


def plot_scurve(input_file: str, output_file: str, title: str) -> None:
    # 1. Setup Theme
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 2. Load Data
    data = pd.read_csv(input_file)
    data['Bandwidth (GB/s)'] = data['throughput'] * 1e6 / (1024 ** 3)  # Convert to GB/s
    data = data[data["impl"] != 'NCCL']  # remove NCCL as its too fast
    
    # 3. Plot All Implementations Together
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x='input_bytes',
        y='Bandwidth (GB/s)',
        hue='impl',
        markers=True,
        dashes=True,
        linewidth=2.5,
        markersize=8,
        palette="viridis"
    )
    
    plt.xscale('log')
    ticks = [
        1024,
        1024*8,
        1024*8*8,
        1024*8*8*8,
        1024*8*8*8*8,
        1024*8*8*8*8*8,
        1024*8*8*8*8*8*8,
        1024*8*8*8*8*8*8*8,
        1024*8*8*8*8*8*8*8*4,
    ]
    plt.gca().set_xticks(ticks)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    
    plt.grid(True, which="minor", ls="--", alpha=0.3)
    plt.grid(True, which="major", ls="-", alpha=0.8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Input Size", fontsize=12)
    plt.ylabel("Per-GPU Bandwidth (GB/s)", fontsize=12)
    plt.legend(title='Implementation', loc="upper left")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot S-Curve from benchmark CSV")
    parser.add_argument("input_file", type=str, help="File containing CSV results")
    parser.add_argument("--output", type=str, default="s_curve.png", help="Output filename")
    parser.add_argument("--title", type=str, default="S-Curve Plot", help="Title of the plot")
    
    args = parser.parse_args()
    plot_scurve(args.input_file, args.output, args.title)