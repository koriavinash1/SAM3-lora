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
    
    def get_class_ids(self) -> list[int]:
        """Get all unique class IDs in this image's annotations."""
        class_ids = set()
        for ann in self.annotations:
            if "category_id" in ann:
                class_ids.add(int(ann["category_id"]))
        return sorted(list(class_ids))


class CocoSplitDataset:
    def __init__(self, coco_json: Path, img_root: Path, split: str, val_split_ratio: float = 0.1) -> None:
        self.coco_json = Path(coco_json)
        self.img_root = Path(img_root)
        self.split = split
        self.val_split_ratio = val_split_ratio
        self._records = self._load_records()
        self._class_stats = None

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

            image_path = self.img_root / (str(image["file_name"]) + '.jpg')
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
    
    def get_class_statistics(self) -> dict:
        """Compute class statistics from all annotations."""
        if self._class_stats is not None:
            return self._class_stats
        
        class_counts = {}
        class_image_counts = {}
        total_annotations = 0
        
        for record in self._records:
            classes_in_image = set()
            for ann in record.annotations:
                class_id = int(ann.get("category_id", -1))
                if class_id >= 0:
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    classes_in_image.add(class_id)
                    total_annotations += 1
            
            for class_id in classes_in_image:
                class_image_counts[class_id] = class_image_counts.get(class_id, 0) + 1
        
        # Compute weights
        class_weights = {}
        if class_counts:
            for class_id, count in class_counts.items():
                # Inverse weight
                weight = 1.0 / (count + 1e-8)
                class_weights[class_id] = weight
            
            # Normalize weights to have mean 1
            if class_weights:
                mean_weight = sum(class_weights.values()) / len(class_weights)
                class_weights = {k: v / mean_weight for k, v in class_weights.items()}
        
        self._class_stats = {
            "class_counts": class_counts,
            "class_image_counts": class_image_counts,
            "class_weights": class_weights,
            "total_annotations": total_annotations,
            "total_images": len(self._records),
        }
        
        return self._class_stats
    
    def split_train_val(self) -> tuple[CocoSplitDataset, CocoSplitDataset]:
        """
        Split dataset into train and validation subsets.
        
        Returns:
            (train_dataset, val_dataset) tuple
        """
        import random
        from copy import copy
        
        # Split records
        num_val = max(1, int(len(self._records) * self.val_split_ratio))
        val_indices = set(random.sample(range(len(self._records)), num_val))
        
        train_records = [r for i, r in enumerate(self._records) if i not in val_indices]
        val_records = [r for i, r in enumerate(self._records) if i in val_indices]
        
        # Create new dataset instances with split records
        train_dataset = copy(self)
        train_dataset._records = train_records
        train_dataset._class_stats = None
        
        val_dataset = copy(self)
        val_dataset._records = val_records
        val_dataset._class_stats = None
        
        return train_dataset, val_dataset
