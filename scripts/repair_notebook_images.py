"""Embed figures in all saved notebooks; optional Sharp rasterizes SVG-only plots."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pinn_pca.notebook_images import embed_notebook_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="node")
    parser.add_argument("--sharp-module", help="Path to an installed sharp Node module, for SVG-only legacy exports")
    args = parser.parse_args()

    def rasterize(raw, path):
        javascript = ("const fs=require('fs'); const sharp=require(process.argv[1]); "
                      "sharp(fs.readFileSync(0),{density:144}).flatten({background:'#ffffff'}).png().toFile(process.argv[2])"
                      ".catch(e=>{console.error(e);process.exit(1)});")
        subprocess.run([args.node, "-e", javascript, args.sharp_module, str(path)], input=raw, check=True, capture_output=True)

    notebooks = sorted(ROOT.glob("pca_*/notebooks/**/*.ipynb"))
    total = 0
    for notebook in notebooks:
        count = embed_notebook_images(notebook, ROOT, rasterize if args.sharp_module else None)
        total += count
        print(f"{notebook.relative_to(ROOT)}: {count} embedded PNG figures", flush=True)
    print(f"Total: {total} figures in {len(notebooks)} notebooks")


if __name__ == "__main__":
    main()
