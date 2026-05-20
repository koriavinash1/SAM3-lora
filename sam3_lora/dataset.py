from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CocoImageRecord:
    image_id: int
    image_path: Path
    width: int
    height: int
    split: str
    annotations: list[dict]


class CocoSplitDataset:
    def __init__(self, coco_json: Path, img_root: Path, split: str) -> None:
        self.coco_json = Path(coco_json)
        self.img_root = Path(img_root)
        self.split = split
        self._records = self._load_records()

    def _annotation_split(self, ann: dict) -> str | None:
        attributes = ann.get("attributes")
        if isinstance(attributes, dict):
            split = attributes.get("split")
            if split is not None:
                return str(split)
        return None

    def _load_records(self) -> list[CocoImageRecord]:
        payload = json.loads(self.coco_json.read_text())

        image_by_id: dict[int, dict] = {int(img["id"]): img for img in payload.get("images", [])}
        annotations_by_image: dict[int, list[dict]] = {}
        for ann in payload.get("annotations", []):
            image_id = int(ann["image_id"])
            annotations_by_image.setdefault(image_id, []).append(ann)

        records: list[CocoImageRecord] = []
        for image_id, image in image_by_id.items():
            image_split = image.get("split")
            split = str(image_split) if image_split is not None else None
            if split is None:
                anns = annotations_by_image.get(image_id, [])
                split = next((s for s in (self._annotation_split(a) for a in anns) if s), "train")

            if split != self.split:
                continue

            image_path = self.img_root / image["file_name"]
            records.append(
                CocoImageRecord(
                    image_id=image_id,
                    image_path=image_path,
                    width=int(image["width"]),
                    height=int(image["height"]),
                    split=split,
                    annotations=annotations_by_image.get(image_id, []),
                )
            )

        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> CocoImageRecord:
        return self._records[index]

    @property
    def records(self) -> list[CocoImageRecord]:
        return self._records
