# Assignment and NMS: deterministic toy demonstration

**Synthetic demonstration only. No detector inference, training, or benchmark evaluation was run.**

Two ground-truth objects and three same-class candidate boxes are constructed by hand. Predictions 0 and 1 overlap the first object; prediction 2 matches the second.

| Prediction | IoU with target 0 | IoU with target 1 | Score |
|---|---:|---:|---:|
| 0 | 1.000000 | 0.000000 | 0.95 |
| 1 | 0.818182 | 0.000000 | 0.90 |
| 2 | 0.000000 | 1.000000 | 0.85 |

Dense toy IoU matching (IoU >= 0.5): [{'prediction': 0, 'target': 0}, {'prediction': 1, 'target': 0}, {'prediction': 2, 'target': 1}].
Minimum-cost one-to-one matching (toy cost 1-IoU): [{'prediction': 0, 'target': 0}, {'prediction': 2, 'target': 1}]; unmatched prediction: [1].
Greedy same-class NMS at IoU 0.5 keeps indices [0, 2].

The dense rule has two positive predictions for target 0. One-to-one matching keeps only one matched prediction per target. In DETR, unmatched queries then receive no-object classification supervision.

Crowding counterexample: two distinct object boxes have IoU 0.666667; the same NMS rule keeps only [0]. Thus geometric suppression can remove a real neighboring object.

Real Faster R-CNN, FCOS, YOLO11, and DETR use different assignment rules and losses. This tiny example isolates one-to-many versus one-to-one cardinality and the consequence of NMS. It does not reproduce those complete algorithms.

Assertions verify the hand-calculated expected results. No randomness or accelerator is used.
