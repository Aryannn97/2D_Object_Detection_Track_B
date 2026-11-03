# 2D Object Detection: A Source-Code Study

**Intelligent Autonomous Systems homework - Track B**  
**Author:** Aryan Dandwekar

This project traces four open-source detectors from image transformation to final detections. Its central question is how **target assignment connects training losses, prediction representation and duplicate removal**.

## Submission

- 10-page technical report: submitted separately as a PDF (not included in this repository)
- [Editable report text](docs/report.md)
- [63 specific source-code references](docs/source_evidence.md), each with an immutable GitHub link, file, symbol, line span, explanation and significance
- [Executed source audit](artifacts/source_audit.md) and [machine-readable hashes/results](artifacts/source_audit.json)
- [Executed CPU assignment/NMS demonstration](artifacts/toy_assignment.md)
- [Reproduction command log](artifacts/run_log.txt)

This is **code and architecture analysis**, not a detector benchmark. No training, pretrained model inference, mAP, latency or GPU-memory measurements are claimed. Toy outputs and statically derived tensor shapes are explicitly labeled.

## Models and source snapshots

| Configuration | Family | Inspected implementation |
|---|---|---|
| `fasterrcnn_resnet50_fpn` (not v2) | Two-stage, anchor-based | [TorchVision v0.20.1](https://github.com/pytorch/vision/tree/3ac97aa9120137381ed1060f37237e44485ac2aa) |
| `fcos_resnet50_fpn` | Dense one-stage, anchor-free | [TorchVision v0.20.1](https://github.com/pytorch/vision/tree/3ac97aa9120137381ed1060f37237e44485ac2aa) |
| YOLO11n, `yolo11.yaml` scale `n` | Dense one-stage, anchor-free | [Ultralytics v8.3.0](https://github.com/ultralytics/ultralytics/tree/6e43d1e1e5db72afbf686dee6745669bcb124b0a) |
| DETR-R50, 100 queries | Query-based set prediction | [Official DETR snapshot](https://github.com/facebookresearch/detr/tree/29901c51d7fe8712168b8d0d64351170bc0f83e0) |

Full commits are locked in [sources.lock.json](sources.lock.json). These versions were deliberately selected for inspectable architectural comparison; this repository does not claim they are today's best-performing detectors.

## Key implementation findings

- FCOS remains anchor-free even though TorchVision uses `AnchorGenerator` as a center/stride coding helper. Its positives come from geometric point rules, not template IoU.
- Faster R-CNN performs anchor matching followed by proposal matching. RPN NMS is grouped by feature level, while final NMS is grouped by class.
- Standard YOLO11 uses task-aligned one-to-many assignment and class-aware NMS. Its classification objective is BCE, with CIoU and DFL for boxes. Nearby end-to-end and varifocal branches are not its active path.
- Original DETR uses a single final backbone scale, Hungarian matching for supervised loss and no NMS in `PostProcess`. Matching uses negative class probability, whereas classification training uses cross entropy.
- One-to-one matching encourages unique predictions; it does not mathematically guarantee that a trained detector can never emit duplicates.

## Reproduce the investigation

Requirements for source checks: **Python 3.10+**, Git, and network access to GitHub. No GPU, torch installation, datasets or weights are required.

```bash
python scripts/fetch_sources.py
python scripts/inspect_sources.py
python scripts/toy_assignment_demo.py
```

The fetcher places source checkouts in `../upstream/`. It checks out the exact locked commits. It can switch a clean, matching-origin checkout to the locked commit, and refuses a different origin or local modifications.

To regenerate the PDF locally (written to `report/`, which is git-ignored):

```bash
python -m pip install -r requirements.txt
python scripts/build_report.py
```

The PDF builder expects DejaVu Sans/Mono at `/usr/share/fonts/truetype/dejavu/`, available on Ubuntu through `fonts-dejavu-core`. It creates the PDF, editable Markdown and four original SVG diagrams. All checks and PDF generation can also be run and recorded together:

```bash
python scripts/run_checks.py
```

Optional PDF visual inspection with Poppler:

```bash
pdftoppm -scale-to 1200 -png report/2D_Object_Detection_Track_B.pdf preview
```

## Repository contents

| Location | Purpose |
|---|---|
| `report/` | Generated PDF output (git-ignored; submitted separately) |
| `docs/report.md` | Editable report text |
| `docs/model_notes/` | Detailed TorchVision, YOLO11 and DETR call-path analysis |
| `docs/figures/` | Four original code-derived SVG diagrams |
| `docs/evidence.json` | Structured code evidence |
| `docs/source_evidence.md` | Human-readable evidence with immutable links |
| `docs/tutorial_provenance.json` | Hash of the course-provided tutorial snapshot |
| `docs/assignment_checklist.md` | Track B requirement-to-deliverable mapping |
| `scripts/` | Fetcher, AST/hash auditor, toy demonstration, report builder and check runner |
| `artifacts/` | Actual audit results, toy outputs, command log and validation results |

## Technical references and AI disclosure

The report cites **nine original research papers**, **three official implementation repositories**, official YOLO11 documentation and the course's `ngdet` tutorial. YOLO11 has no formal original paper according to its official documentation; the report does not invent one. TOOD/GFL references explain component ideas rather than establish that YOLO11 is an exact implementation of either paper.

AI tools assisted navigation, explanation, scripts, diagrams and drafting. Findings were checked against the pinned source. The audit validates source identity and named locations, not every semantic interpretation. The toy demonstration illustrates principles rather than reproducing a detector's full loss or training dynamics. No model evaluation was performed.

Upstream source and weights are not redistributed. See [third-party notices](THIRD_PARTY_NOTICES.md).
