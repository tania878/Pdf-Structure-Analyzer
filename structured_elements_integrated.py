# Structured PDF element data model
# Save beside pdf_to_cells.py as structured_elements.py.
# This module organizes extracted results; it does not extract the PDF itself.

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from uuid import uuid4
import json


def _new_id():
    return "el_" + uuid4().hex[:10]


@dataclass
class Element:
    type: str
    content: Any = ""
    page: int = 1
    bbox: list | None = None
    id: str = field(default_factory=_new_id)
    style: dict = field(default_factory=lambda: {
        "font_family": "Helvetica",
        "font_size": 11,
        "bold": False,
        "italic": False,
        "underline": False,
        "alignment": "left",
        "text_color": "#000000",
    })
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.type = self.type.strip().lower()
        if self.page < 1:
            raise ValueError("page must be 1 or greater")
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox must be [x0, y0, x1, y1]")
            self.bbox = [float(v) for v in self.bbox]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data.get("id", _new_id()),
            type=data.get("type", "other"),
            content=data.get("content", ""),
            page=int(data.get("page", 1)),
            bbox=data.get("bbox"),
            style=dict(data.get("style") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


class StructuredDocument:
    def __init__(self, source_pdf="", type_styles=None):
        self.source_pdf = source_pdf
        self.elements = []
        # Shared defaults apply to every element of the same type.
        self.type_styles = dict(type_styles or {})

    def update_type_style(self, element_type, **changes):
        element_type = element_type.strip().lower()
        self.type_styles.setdefault(element_type, {}).update(changes)
        return True

    def get_effective_style(self, element):
        return {**self.type_styles.get(element.type, {}), **element.style}

    def add(self, type, content="", page=1, bbox=None,
            style=None, metadata=None, element_id=None):
        element = Element(
            id=element_id or _new_id(),
            type=type,
            content=content,
            page=int(page or 1),
            bbox=bbox,
            style=dict(style) if style is not None else {},
            metadata=dict(metadata) if metadata is not None else {},
        )
        self.elements.append(element)
        return element

    def add_extracted_item(self, item):
        """Convert one dictionary from your current extractor into an Element."""
        item = dict(item)
        kind = item.pop("type", None) or item.pop("kind", None) or item.pop("category", "other")
        if kind == "image":
            content = item.pop("content", item.pop("src", item.pop("value", "")))
        elif kind == "table":
            content = item.pop("content", {
                "header": item.pop("header", []),
                "rows": item.pop("rows", [])
            })
        else:
            content = item.pop("content", item.pop("text", item.pop("value", "")))
        page = item.pop("page", item.pop("page_number", 1))
        bbox = item.pop("bbox", item.pop("rect", item.pop("position", None)))
        element_id = item.pop("id", None)
        if isinstance(element_id, int):
            element_id = f"el_{element_id:04d}"   # stable ids: same PDF -> same ids

        style = dict(item.pop("style", {}) or {})
        aliases = {
            "font": "font_family",
            "font_name": "font_family",
            "font_family": "font_family",
            "font_size": "font_size",
            "bold": "bold",
            "italic": "italic",
            "underline": "underline",
            "alignment": "alignment",
            "text_color": "text_color",
        }
        for key, target in aliases.items():
            if key in item:
                style[target] = item.pop(key)

        metadata = dict(item.pop("metadata", {}) or {})
        metadata.update(item)
        return self.add(kind, content, page, bbox, style, metadata, element_id)

    def get(self, element_id):
        return next((e for e in self.elements if e.id == element_id), None)

    def update_style(self, element_id, **changes):
        element = self.get(element_id)
        if element is None:
            return False
        element.style.update(changes)
        return True

    def remove(self, element_id):
        for i, element in enumerate(self.elements):
            if element.id == element_id:
                del self.elements[i]
                return True
        return False

    def to_dict(self):
        return {
            "source_pdf": self.source_pdf,
            "element_count": len(self.elements),
            "type_styles": self.type_styles,
            "elements": [e.to_dict() for e in self.elements],
        }

    def save_json(self, output_path="structured_elements.json"):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load_json(cls, input_path):
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        doc = cls(data.get("source_pdf", ""), data.get("type_styles", {}))
        doc.elements = [Element.from_dict(item) for item in data.get("elements", [])]
        return doc


def build_document(extracted_items, source_pdf=""):
    """Build a StructuredDocument from a list of extraction dictionaries."""
    doc = StructuredDocument(source_pdf)
    for item in extracted_items:
        doc.add_extracted_item(item)
    return doc


if __name__ == "__main__":
    # Self-test with sample content. This does not process a real PDF.
    doc = StructuredDocument("sample_purchase_order.pdf")
    heading = doc.add(
        "heading", "Purchase Order", page=1, bbox=[50, 80, 500, 115],
        style={"font_family": "Helvetica", "font_size": 24, "bold": True,
               "alignment": "center"}
    )
    doc.add("subheading", "Supplier Details", page=1, bbox=[50, 130, 300, 155])
    doc.add("paragraph", "Supplier information goes here.",
            page=1, bbox=[50, 165, 500, 205])
    doc.add("image", "assets/image_001.png", page=1, bbox=[100, 220, 350, 400],
            metadata={"keep_aspect_ratio": True})
    doc.add("link", "Supplier website", page=1, bbox=[50, 420, 250, 440],
            metadata={"url": "https://example.com"})
    doc.add("table", [["Item", "Quantity"], ["Pen", "10"]],
            page=1, bbox=[50, 450, 500, 550])
    doc.update_style(heading.id, font_size=28, alignment="left")
    output = doc.save_json("structured_elements.json")
    print(f"Saved {len(doc.elements)} sample elements to {output.resolve()}")
    print("Sample only: connect this model to pdf_to_cells.py to use real PDF data.")
