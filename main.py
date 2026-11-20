from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fractals


def load_max_depth(config_path, default_depth=6):
    """Read max recursion depth from a relative config file with error handling."""
    try:
        text = config_path.read_text(encoding="utf-8").strip()
        return int(text)
    except FileNotFoundError:
        print(f"Info: '{config_path.name}' not found. Using default max_depth = {default_depth}.")
        return default_depth
    except ValueError:
        print(f"Warning: '{config_path.name}' is invalid. Using default max_depth = {default_depth}.")
        return default_depth


def compute_metrics_table(max_depth):
    """Build a Pandas DataFrame comparing Koch Snowflake and Sierpiński metrics."""
    rows = []
    for d in range(max_depth + 1):
        # Koch Snowflake metrics
        koch_pts = fractals.koch_snowflake(d)
        k_perim = fractals.polygon_perimeter(koch_pts)
        k_area = fractals.polygon_area(koch_pts)

        # Sierpiński Triangle metrics
        tris = fractals.sierpinski_triangles(d)
        s_perim = sum(fractals.polygon_perimeter(np.vstack([t, t[0]])) for t in tris)
        s_area = sum(fractals.polygon_area(np.vstack([t, t[0]])) for t in tris)

        rows.append({
            "depth": d,
            "koch_segments": len(koch_pts) - 1,
            "koch_perimeter_units": round(k_perim, 6),
            "koch_area_sq_units": round(k_area, 6),
            "sierpinski_triangles": len(tris),
            "sierpinski_perimeter_units": round(s_perim, 6),
            "sierpinski_area_sq_units": round(s_area, 6),
        })
    return pd.DataFrame(rows)


def save_comparison_plot(df, fig_path):
    """Plot and save Perimeter and Area vs. Depth with clear labels, units, and legends."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Perimeter vs Depth
    ax1.plot(df["depth"], df["koch_perimeter_units"], marker="o", linewidth=2, label="Koch Snowflake")
    ax1.plot(df["depth"], df["sierpinski_perimeter_units"], marker="^", linewidth=2, label="Sierpiński Triangle")
    ax1.set_title("Fractal Perimeter vs. Recursion Depth")
    ax1.set_xlabel("Recursion Depth (iterations)")
    ax1.set_ylabel("Total Perimeter (units)")
    ax1.set_xticks(df["depth"])
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: Area vs Depth
    ax2.plot(df["depth"], df["koch_area_sq_units"], marker="o", linewidth=2, label="Koch Snowflake")
    ax2.plot(df["depth"], df["sierpinski_area_sq_units"], marker="^", linewidth=2, label="Sierpiński Triangle")
    ax2.set_title("Fractal Area vs. Recursion Depth")
    ax2.set_xlabel("Recursion Depth (iterations)")
    ax2.set_ylabel("Enclosed Area (square units)")
    ax2.set_xticks(df["depth"])
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # Relative paths anchored to the project folder
    project_dir = Path(__file__).resolve().parent
    config_file = project_dir / "config.txt"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_output = output_dir / "fractal_metrics.csv"
    png_output = output_dir / "fractal_comparison.png"

    max_depth = load_max_depth(config_file, default_depth=6)
    metrics_df = compute_metrics_table(max_depth)

    # Save results table (.csv) and figure (.png)
    metrics_df.to_csv(csv_output, index=False)
    save_comparison_plot(metrics_df, png_output)

    print(metrics_df.to_string(index=False))
    print(f"\nSaved table to:  {csv_output.relative_to(project_dir)}")
    print(f"Saved figure to: {png_output.relative_to(project_dir)}")

    