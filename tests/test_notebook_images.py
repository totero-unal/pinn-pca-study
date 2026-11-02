import base64
import json
import tempfile
import unittest
from pathlib import Path

from pinn_pca.notebook_images import embed_notebook_images


PIXEL = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=")


class NotebookImageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "pca_collocation/notebooks/demo.ipynb"
        self.path.parent.mkdir(parents=True)
        self.asset = self.root / "pca_collocation/results/chart.png"
        self.asset.parent.mkdir(parents=True)
        self.asset.write_bytes(PIXEL)

    def save(self, cells):
        self.path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")

    def test_markdown_figures_are_embedded_in_order_and_repair_is_idempotent(self):
        self.save([{"cell_type": "markdown", "metadata": {}, "source": "Before\n![First](../results/chart.png)\nMiddle\n![Second](../results/chart.png)\nAfter"}])
        self.assertEqual(embed_notebook_images(self.path, self.root), 2)
        cells = json.loads(self.path.read_text(encoding="utf-8"))["cells"]
        figures = [c for c in cells if c["cell_type"] == "code"]
        self.assertEqual(len(figures), 2)
        for cell in figures:
            self.assertEqual(base64.b64decode(cell["outputs"][0]["data"]["image/png"]), PIXEL)
            self.assertIn("embed=True", cell["source"])
            compile(cell["source"], "figure", "exec")
        self.assertEqual(cells[-1]["source"], "After")
        first = self.path.read_bytes()
        embed_notebook_images(self.path, self.root)
        self.assertEqual(first, self.path.read_bytes())

    def test_svg_fallback_is_saved_without_touching_canonical_svg(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>'
        source = self.asset.with_suffix(".svg")
        source.write_text(svg, encoding="utf-8")
        self.asset.unlink()
        self.save([{"cell_type": "code", "metadata": {}, "execution_count": 1,
                    "source": "display(SVG(filename='chart.svg'))", "outputs": [{"output_type": "display_data", "metadata": {}, "data": {"image/svg+xml": svg}}]}])
        calls = []
        def converter(raw, destination):
            calls.append(raw)
            destination.write_bytes(PIXEL)
        embed_notebook_images(self.path, self.root, converter)
        self.assertEqual(calls, [svg.encode()])
        self.assertEqual(source.read_text(encoding="utf-8"), svg)
        data = json.loads(self.path.read_text(encoding="utf-8"))["cells"][0]["outputs"][0]["data"]
        self.assertIn("image/png", data)
        self.assertNotIn("image/svg+xml", data)

    def test_missing_image_leaves_notebook_untouched(self):
        self.save([{"cell_type": "markdown", "metadata": {}, "source": "![Missing](missing.png)"}])
        before = self.path.read_bytes()
        with self.assertRaises(FileNotFoundError):
            embed_notebook_images(self.path, self.root)
        self.assertEqual(before, self.path.read_bytes())

    def test_nonfigure_code_and_outputs_are_preserved(self):
        cell = {"cell_type": "code", "metadata": {}, "execution_count": 3,
                "source": "x = 2\nprint(x)", "outputs": [{"output_type": "stream", "name": "stdout", "text": "2\n"}]}
        self.save([cell])
        embed_notebook_images(self.path, self.root)
        saved = json.loads(self.path.read_text(encoding="utf-8"))["cells"][0]
        for key, value in cell.items():
            self.assertEqual(saved[key], value)


if __name__ == "__main__":
    unittest.main()
