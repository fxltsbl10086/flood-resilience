"""Render supplementary figures S1–S6 from archived inputs."""
import argparse
from pathlib import Path
from _release import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    ids = [f"S{i}" for i in range(1, 7)]
    parser.add_argument("figures", nargs="*", choices=ids, metavar="FIGURE")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results/figures/rebuilt")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for figure in args.figures or ids:
        if figure in ("S5", "S6"):
            from _figure_s5_s6 import run
        else:
            from _figure_s1_s4 import run
        run(figure, output)


if __name__ == "__main__":
    main()

