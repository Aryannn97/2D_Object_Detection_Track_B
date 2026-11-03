#!/usr/bin/env python3
"""Tiny deterministic demonstrations of assignment and greedy NMS.

These hand-constructed cases illustrate ideas. They are not detector outputs,
benchmarks, training runs, or numerical reproductions of any paper's results.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def greedy_nms(boxes: list[list[float]], scores: list[float], threshold: float) -> list[int]:
    remaining = sorted(range(len(boxes)), key=lambda index: (-scores[index], index))
    kept = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [index for index in remaining if iou(boxes[best], boxes[index]) <= threshold]
    return kept


def one_to_one(cost: list[list[float]]) -> dict:
    """Exhaustive injective assignment for three predictions and two GT objects.

    Equivalent objective to a rectangular minimum-cost assignment, solved by
    enumeration for transparency. Real DETR uses scipy's Hungarian solver and
    a cost combining class probability, L1 distance, and generalized IoU.
    """
    targets = len(cost[0])
    assignments = []
    for prediction_indices in itertools.permutations(range(len(cost)), targets):
        total = sum(cost[prediction_indices[target]][target] for target in range(targets))
        assignments.append((total, prediction_indices))
    total, indices = min(assignments)
    return {"matches": [{"prediction": pred, "target": target} for target, pred in enumerate(indices)],
            "total_cost": total, "unmatched_predictions": sorted(set(range(len(cost))) - set(indices)),
            "number_of_assignments_enumerated": len(assignments)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
    predictions = [[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10]]
    scores = [0.95, 0.90, 0.85]
    overlaps = [[iou(box, target) for target in gt] for box in predictions]
    dense = [{"prediction": p, "target": t} for p, row in enumerate(overlaps) for t, overlap in enumerate(row) if overlap >= 0.5]
    # Deliberately simple 1-IoU cost. DETR's actual matching cost is richer.
    cost = [[1 - overlap for overlap in row] for row in overlaps]
    matching = one_to_one(cost)
    kept = greedy_nms(predictions, scores, 0.5)
    # Two nearby real objects can suppress each other: NMS has no GT knowledge.
    crowded = [[0, 0, 10, 10], [2, 0, 12, 10]]
    crowded_kept = greedy_nms(crowded, [0.9, 0.8], 0.5)
    result = {"kind": "synthetic_toy_demonstration_not_model_execution", "randomness": "none",
              "coordinate_convention": "continuous xyxy; no inclusive +1 pixel convention",
              "ground_truth": gt, "predictions": predictions, "same_class_scores": scores,
              "prediction_target_iou": overlaps, "toy_dense_iou_threshold": 0.5,
              "toy_dense_positive_pairs": dense, "toy_matching_cost": "1-IoU; not DETR's exact cost",
              "one_to_one_assignment": matching, "greedy_nms_threshold": 0.5, "greedy_nms_kept_indices": kept,
              "crowding_case": {"two_distinct_object_boxes": crowded, "iou": iou(*crowded), "nms_kept_indices": crowded_kept},
              "interpretation": ["A dense positive rule can supervise more than one prediction for one target.",
                                  "A one-to-one match selects at most one prediction per target; DETR supervises unmatched queries as no-object.",
                                  "Greedy NMS removes nearby same-class duplicates, but can also suppress distinct overlapping objects.",
                                  "No-object supervision encourages, but does not mathematically guarantee, duplicate-free DETR inference."]}
    assert dense == [{"prediction": 0, "target": 0}, {"prediction": 1, "target": 0}, {"prediction": 2, "target": 1}]
    assert matching["matches"] == [{"prediction": 0, "target": 0}, {"prediction": 2, "target": 1}]
    assert kept == [0, 2] and crowded_kept == [0]
    assert math.isclose(overlaps[1][0], 9 / 11)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "toy_assignment.json").write_text(json.dumps(result, indent=2) + "\n")
    md = ["# Assignment and NMS: deterministic toy demonstration", "", "**Synthetic demonstration only. No detector inference, training, or benchmark evaluation was run.**", "",
          "Two ground-truth objects and three same-class candidate boxes are constructed by hand. Predictions 0 and 1 overlap the first object; prediction 2 matches the second.", "",
          "| Prediction | IoU with target 0 | IoU with target 1 | Score |", "|---|---:|---:|---:|"]
    md += [f"| {index} | {row[0]:.6f} | {row[1]:.6f} | {scores[index]:.2f} |" for index, row in enumerate(overlaps)]
    md += ["", f"Dense toy IoU matching (IoU >= 0.5): {dense}.",
           f"Minimum-cost one-to-one matching (toy cost 1-IoU): {matching['matches']}; unmatched prediction: {matching['unmatched_predictions']}.",
           f"Greedy same-class NMS at IoU 0.5 keeps indices {kept}.", "",
           "The dense rule has two positive predictions for target 0. One-to-one matching keeps only one matched prediction per target. In DETR, unmatched queries then receive no-object classification supervision.", "",
           f"Crowding counterexample: two distinct object boxes have IoU {iou(*crowded):.6f}; the same NMS rule keeps only {crowded_kept}. Thus geometric suppression can remove a real neighboring object.", "",
           "Real Faster R-CNN, FCOS, YOLO11, and DETR use different assignment rules and losses. This tiny example isolates one-to-many versus one-to-one cardinality and the consequence of NMS. It does not reproduce those complete algorithms.", "",
           "Assertions verify the hand-calculated expected results. No randomness or accelerator is used.", ""]
    (args.output_dir / "toy_assignment.md").write_text("\n".join(md))
    print("Toy assertions passed: 3 dense positives; 2 one-to-one matches; NMS keeps [0, 2]; crowded case keeps [0].")


if __name__ == "__main__":
    main()
