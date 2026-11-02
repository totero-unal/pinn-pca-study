"""Embed PNG figure outputs so saved notebooks do not depend on SVG or URLs."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
from pathlib import Path
from urllib.parse import unquote

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def png_output(path):
    path = Path(path)
    raw = path.read_bytes()
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError(f"Not a PNG figure: {path}")
    width, height = struct.unpack(">II", raw[16:24])
    return {"output_type": "display_data", "metadata": {}, "data": {
        "image/png": base64.b64encode(raw).decode("ascii"),
        "text/plain": f"{path.name} ({width} × {height})",
    }}


def _figure_cell(paths, root, metadata=None):
    relative = [p.resolve().relative_to(root).as_posix() for p in paths]
    source = ("from pathlib import Path\nfrom IPython.display import Image, display\n"
              "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').is_file())\n")
    for item in relative:
        source += f"display(Image(filename=str(ROOT / {item!r}), format='png', embed=True))\n"
    return {"cell_type": "code", "execution_count": None,
            "metadata": {**(metadata or {}), "pinn_figure_assets": relative},
            "source": source, "outputs": [png_output(p) for p in paths]}


def embed_notebook_images(path, root, rasterize_svg=None):
    """Migrate local Markdown images and saved SVG outputs to saved PNG outputs.

    Existing PNG exports are reused by matching the exact saved SVG content.
    Older SVG-only exports require a caller-provided rasterizer, once. Canonical
    SVG files remain available. Figure cells can then run using standard IPython.
    """
    path, root = Path(path).resolve(), Path(root).resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    original = json.dumps(notebook, ensure_ascii=False, sort_keys=True)
    svg_index = None

    def from_svg(svg):
        nonlocal svg_index
        raw = ''.join(svg).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        if svg_index is None:
            svg_index = {}
            for folder in root.glob("pca_*/results"):
                for candidate in folder.rglob("*.svg"):
                    svg_index[hashlib.sha256(candidate.read_bytes()).hexdigest()] = candidate
        candidate = svg_index.get(digest)
        png = candidate.with_suffix(".png") if candidate else path.parent / "_figures" / f"svg_{digest[:20]}.png"
        if not png.is_file():
            if rasterize_svg is None:
                raise FileNotFoundError(f"PNG fallback missing: {png}. Run scripts/repair_notebook_images.py with a rasterizer.")
            png.parent.mkdir(parents=True, exist_ok=True)
            rasterize_svg(raw, png)
        return png

    cells, figures = [], 0
    for cell in notebook["cells"]:
        source = ''.join(cell["source"])
        if cell["cell_type"] == "markdown" and MARKDOWN_IMAGE.search(source):
            cursor = 0
            for match in MARKDOWN_IMAGE.finditer(source):
                target = unquote(match[2].strip("<>"))
                if "://" in target or target.startswith(("data:", "attachment:")):
                    raise ValueError(f"Expected a local figure in {path}: {target}")
                image_path = (path.parent / target).resolve()
                image_path.relative_to(root)
                text = source[cursor:match.start()].strip()
                if match[1]:
                    text += f"\n\n**{match[1]}**"
                if text.strip():
                    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()})
                cells.append(_figure_cell([image_path], root))
                figures += 1
                cursor = match.end()
            if source[cursor:].strip():
                cells.append({"cell_type": "markdown", "metadata": {}, "source": source[cursor:].strip()})
        elif cell["cell_type"] == "code" and cell.get("metadata", {}).get("pinn_figure_assets"):
            paths = [root / p for p in cell["metadata"]["pinn_figure_assets"]]
            cells.append(_figure_cell(paths, root, cell["metadata"]))
            figures += len(paths)
        elif cell["cell_type"] == "code" and any("image/svg+xml" in o.get("data", {}) for o in cell.get("outputs", [])):
            paths = [from_svg(o["data"]["image/svg+xml"]) for o in cell["outputs"] if "image/svg+xml" in o.get("data", {})]
            if any(o.get("output_type") not in ("display_data", "execute_result") or "image/svg+xml" not in o.get("data", {}) for o in cell["outputs"]):
                raise ValueError(f"Mixed figure and non-figure outputs require manual migration: {path}")
            cells.append(_figure_cell(paths, root, cell.get("metadata")))
            figures += len(paths)
        else:
            cells.append(cell)
            figures += sum("image/png" in o.get("data", {}) for o in cell.get("outputs", []))
    # Cell IDs are mandatory since notebook format 4.5; assign deterministic IDs.
    for index, cell in enumerate(cells):
        cell.setdefault("id", hashlib.sha256(f"{index}:{''.join(cell['source'])}".encode()).hexdigest()[:12])
    notebook["cells"] = cells
    notebook.setdefault("metadata", {})["pinn_figure_format"] = "embedded image/png display_data"
    if json.dumps(notebook, ensure_ascii=False, sort_keys=True) != original:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return figures
