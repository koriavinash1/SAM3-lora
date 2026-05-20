from __future__ import annotations

import json

from sam3_lora.dataset import CocoSplitDataset



def test_dataset_uses_image_split(tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 80, "split": "train"},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 80, "split": "val"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 2, "category_id": 1},
        ],
        "categories": [{"id": 1, "name": "fish"}],
    }
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco))

    ds = CocoSplitDataset(coco_json=coco_path, img_root=tmp_path, split="val")
    assert len(ds) == 1
    assert ds[0].image_id == 2



def test_dataset_falls_back_to_annotation_split(tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "attributes": {"split": "test"},
            }
        ],
        "categories": [{"id": 1, "name": "fish"}],
    }
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco))

    ds = CocoSplitDataset(coco_json=coco_path, img_root=tmp_path, split="test")
    assert len(ds) == 1
    assert ds[0].split == "test"
