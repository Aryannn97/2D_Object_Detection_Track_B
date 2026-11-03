# Track B submission checklist

| Assignment requirement | Evidence |
|---|---|
| At least 4 models from at least 3 families | Faster R-CNN (two-stage), FCOS and YOLO11 (dense one-stage), DETR (query-based); PDF p.1 |
| Detector evolution/taxonomy | PDF p.1: evolution paragraph and model/paper/source/family table |
| Real established/official repositories | Full commits in `sources.lock.json`, source audit and references [10]-[12] |
| Paper, framework, main definition, backbone, neck, head, loss and inference | PDF pp.2-5 and `model_notes/`; all four use PyTorch implementations |
| Code-derived pipeline diagram for each detector | Four original SVGs in `figures/` (inference path plus a dashed training-path strip), embedded in PDF pp.2-5; more detailed Mermaid graphs in notes |
| Backbone/neck/head analysis | PDF pp.2-5, comparison p.7 |
| Anchor-based, anchor-free and query-based distinction | PDF pp.1-7, especially FCOS coding-helper distinction |
| Loss and target-assignment analysis | PDF pp.2-6; BCE/CE/focal, Smooth L1, IoU/GIoU/CIoU, DFL and matching are distinguished |
| At least 3 target-assignment strategies | Four strategies on PDF p.6: anchor/IoU, center sampling, task-aligned, Hungarian |
| Confidence, decoding, NMS, top-K and coordinate restoration | PDF pp.2-5 and p.7 |
| Code-level comparison table | PDF p.7 |
| Required deep dive with actual source | YOLO to DETR: supervision cardinality and NMS, PDF p.8 |
| At least 8 specific code references with what/why | Ten concrete examples in PDF p.9; 63 detailed references in `source_evidence.md` |
| At least 8 credible references including 4 original papers and 2 official repos/docs | PDF p.10: nine original papers, three official repositories, course tutorial and official YOLO11 documentation |
| Evidence of actual source inspection | Locked clean clones, real AST definitions, file/span SHA-256 hashes, detailed call-path notes, actual source audit |
| Reproducibility | `scripts/fetch_sources.py`, `inspect_sources.py`, `toy_assignment_demo.py`, `build_report.py`, `run_checks.py`; actual command log |
| One PDF of 6-10 pages | Submitted separately; 10 pages (not stored in this repository) |
| AI tool usage and limitations | PDF p.9 and README disclosure |

The open-vocabulary text-encoder trace requirement is conditional and does not apply: all four selected detectors are closed-set. Track A dataset/metric requirements do not apply to this Track B submission.
