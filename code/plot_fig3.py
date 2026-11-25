"""Render manuscript Fig. 3 and retain the original data-to-glyph checks."""
from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '_support/figures_2_3_s1_s4'))


def run(output: Path, panel: str | None = None) -> Path:
    import fig3_render as plot
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plot.OUTPUT_DIR = output
    curves, connectivity, loss, _ = plot.load_data()
    if panel is None:
        path, _ = plot.draw(curves, connectivity, loss)
    elif panel == 'b':
        path, _ = plot.draw_panel_b(curves, connectivity)
    else:
        raise ValueError()
    print(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', '--output', type=Path,
                        default=HERE.parent / 'results/figures')
    parser.add_argument('--panel', choices=['b'], help='Export healthcare panel b as fig3b.png.')
    args = parser.parse_args()
    run(args.output_dir, panel=args.panel)
