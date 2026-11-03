# Static source-code inspection audit

This audit verifies source locations and identities. It does not run detector inference or training.

Generated: 2026-10-27T18:20:04.252923+00:00
Python: 3.12.14
Evidence checks: 63/63 passed.

## Snapshot identity

| Repository | Commit | Tracked files clean |
|---|---|---|
| vision | `3ac97aa9120137381ed1060f37237e44485ac2aa` | True |
| ultralytics | `6e43d1e1e5db72afbf686dee6745669bcb124b0a` | True |
| detr | `29901c51d7fe8712168b8d0d64351170bc0f83e0` | True |

## Code references

| ID | File and symbol | Lines | Status |
|---|---|---|---|
| TV01 | `torchvision/models/detection/generalized_rcnn.py` / `GeneralizedRCNN.forward` | 46-110 | PASS |
| TV02 | `torchvision/models/detection/faster_rcnn.py` / `_default_anchorgen` | 37-40 | PASS |
| TV03 | `torchvision/models/detection/faster_rcnn.py` / `FasterRCNN.__init__` | 167-283 | PASS |
| TV04 | `torchvision/models/detection/rpn.py` / `RegionProposalNetwork.assign_targets_to_anchors` | 193-229 | PASS |
| TV05 | `torchvision/models/detection/_utils.py` / `Matcher.__call__` | 358-396 | PASS |
| TV06 | `torchvision/models/detection/rpn.py` / `RegionProposalNetwork.compute_loss` | 299-334 | PASS |
| TV07 | `torchvision/models/detection/rpn.py` / `RegionProposalNetwork.filter_proposals` | 242-297 | PASS |
| TV08 | `torchvision/models/detection/roi_heads.py` / `RoIHeads.select_training_samples` | 630-666 | PASS |
| TV09 | `torchvision/models/detection/roi_heads.py` / `fastrcnn_loss` | 12-49 | PASS |
| TV10 | `torchvision/models/detection/roi_heads.py` / `RoIHeads.postprocess_detections` | 668-725 | PASS |
| TV11 | `torchvision/models/detection/backbone_utils.py` / `_resnet_fpn_extractor` | 118-150 | PASS |
| TV12 | `torchvision/ops/feature_pyramid_network.py` / `FeaturePyramidNetwork.forward` | 172-204 | PASS |
| TV13 | `torchvision/models/detection/fcos.py` / `FCOS.__init__` | 365-426 | PASS |
| TV14 | `torchvision/models/detection/fcos.py` / `FCOS.compute_loss` | 440-487 | PASS |
| TV15 | `torchvision/models/detection/fcos.py` / `FCOSHead.compute_loss` | 52-125 | PASS |
| TV16 | `torchvision/models/detection/fcos.py` / `FCOSClassificationHead.forward` | 183-198 | PASS |
| TV17 | `torchvision/models/detection/fcos.py` / `FCOSRegressionHead.forward` | 245-267 | PASS |
| TV18 | `torchvision/models/detection/_utils.py` / `BoxLinearCoder.encode` | 240-273 | PASS |
| TV19 | `torchvision/models/detection/fcos.py` / `FCOS.postprocess_detections` | 489-556 | PASS |
| TV20 | `torchvision/models/detection/transform.py` / `GeneralizedRCNNTransform.postprocess` | 257-277 | PASS |
| TV21 | `torchvision/models/detection/fcos.py` / `fcos_resnet50_fpn` | 766-775 | PASS |
| TV22 | `torchvision/models/resnet.py` / `resnet50` | 736-763 | PASS |
| TV23 | `torchvision/ops/focal_loss.py` / `sigmoid_focal_loss` | 7-58 | PASS |
| TV24 | `torchvision/models/detection/_utils.py` / `Matcher.set_low_quality_matches_` | 398-417 | PASS |
| TV25 | `torchvision/models/detection/_utils.py` / `BoxLinearCoder.decode` | 275-311 | PASS |
| Y01 | `ultralytics/cfg/models/11/yolo11.yaml` / `backbone / head graph` | 15-47 | PASS |
| Y02 | `ultralytics/nn/tasks.py` / `parse_model` | 933-1045 | PASS |
| Y03 | `ultralytics/nn/modules/block.py` / `C3k2.__init__` | 723-731 | PASS |
| Y04 | `ultralytics/nn/modules/block.py` / `SPPF.forward` | 169-188 | PASS |
| Y05 | `ultralytics/nn/modules/block.py` / `C2PSA.forward` | 1037-1041 | PASS |
| Y06 | `ultralytics/nn/modules/head.py` / `Detect` | 21-68 | PASS |
| Y07 | `ultralytics/nn/modules/head.py` / `Detect._inference` | 94-120 | PASS |
| Y08 | `ultralytics/nn/modules/block.py` / `DFL` | 55-73 | PASS |
| Y09 | `ultralytics/nn/tasks.py` / `DetectionModel.init_criterion` | 386-388 | PASS |
| Y10 | `ultralytics/utils/loss.py` / `v8DetectionLoss` | 157-260 | PASS |
| Y11 | `ultralytics/utils/tal.py` / `TaskAlignedAssigner` | 90-125 | PASS |
| Y12 | `ultralytics/utils/tal.py` / `TaskAlignedAssigner.forward` | 72-88 | PASS |
| Y13 | `ultralytics/utils/tal.py` / `TaskAlignedAssigner.select_highest_overlaps` | 235-265 | PASS |
| Y14 | `ultralytics/utils/loss.py` / `DFLoss.__call__` | 73-88 | PASS |
| Y15 | `ultralytics/utils/tal.py` / `make_anchors` | 303-315 | PASS |
| Y16 | `ultralytics/models/yolo/detect/predict.py` / `DetectionPredictor.postprocess` | 23-40 | PASS |
| Y17 | `ultralytics/utils/ops.py` / `non_max_suppression` | 226-293 | PASS |
| Y18 | `ultralytics/utils/ops.py` / `scale_boxes` | 88-123 | PASS |
| Y19 | `ultralytics/engine/predictor.py` / `BasePredictor.preprocess` | 117-133 | PASS |
| Y20 | `ultralytics/utils/loss.py` / `BboxLoss.forward` | 99-113 | PASS |
| Y21 | `ultralytics/utils/tal.py` / `dist2bbox` | 318-327 | PASS |
| Y22 | `ultralytics/engine/predictor.py` / `BasePredictor.pre_transform` | 144-156 | PASS |
| D01 | `models/backbone.py` / `BackboneBase.__init__` | 58-70 | PASS |
| D02 | `models/backbone.py` / `Backbone.__init__` | 83-93 | PASS |
| D03 | `models/detr.py` / `DETR.__init__` | 21-42 | PASS |
| D04 | `models/detr.py` / `DETR.forward` | 59-72 | PASS |
| D05 | `models/transformer.py` / `Transformer.forward` | 47-59 | PASS |
| D06 | `models/transformer.py` / `TransformerDecoderLayer.forward_post` | 212-233 | PASS |
| D07 | `models/matcher.py` / `HungarianMatcher.forward` | 34-82 | PASS |
| D08 | `models/detr.py` / `SetCriterion.loss_labels` | 108-127 | PASS |
| D09 | `models/detr.py` / `SetCriterion.loss_boxes` | 143-162 | PASS |
| D10 | `models/detr.py` / `SetCriterion.forward` | 215-253 | PASS |
| D11 | `models/detr.py` / `PostProcess.forward` | 258-286 | PASS |
| D12 | `engine.py` / `evaluate` | 107-108 | PASS |
| D13 | `main.py` / `get_args_parser` | 35-79 | PASS |
| D14 | `models/detr.py` / `build` | 304-335 | PASS |
| D15 | `util/box_ops.py` / `box_cxcywh_to_xyxy` | 9-13 | PASS |
| D16 | `engine.py` / `train_one_epoch` | 32-35 | PASS |

File and selected-span SHA256 values, claims, reasons, definition locations, and immutable permalinks are in `source_audit.json`.
The script verifies that an evidence span overlaps the named definition; it cannot prove the interpretation of that code is correct.
