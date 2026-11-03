# Source-code evidence index

Each entry connects an actual pinned source location to its role in the detector. All code was read from official cloned repositories. Python definition checks and file SHA-256 values are recorded in [the audit](../artifacts/source_audit.json). The audit proves source identity/location, not semantic correctness.

## TV01: `GeneralizedRCNN.forward`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/generalized_rcnn.py`, lines 46-110](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/generalized_rcnn.py#L46-L110)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Calls image transform, backbone, RPN, RoI heads, coordinate restoration and loss merge.

**Why it matters:** Establishes actual two-stage orchestration rather than an assumed architecture.

## TV02: `_default_anchorgen`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/faster_rcnn.py`, lines 37-40](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/faster_rcnn.py#L37-L40)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Defines five pyramid anchor scales and three aspect ratios per location.

**Why it matters:** Documents shape priors and the number of RPN candidates.

## TV03: `FasterRCNN.__init__`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/faster_rcnn.py`, lines 167-283](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/faster_rcnn.py#L167-L283)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Builds RPN, 7x7 MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor and default hyperparameters.

**Why it matters:** Pins the exact non-v2 variant and both suppression stages.

## TV04: `RegionProposalNetwork.assign_targets_to_anchors`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/rpn.py`, lines 193-229](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/rpn.py#L193-L229)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Forms GT-anchor IoU matrix and labels positives, negatives and ignored anchors.

**Why it matters:** Shows many-to-one anchor matching in executable code.

## TV05: `Matcher.__call__`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/_utils.py`, lines 358-396](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/_utils.py#L358-L396)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Thresholds each candidate's best GT IoU into foreground, background or ignore and invokes optional low-quality matching.

**Why it matters:** Explains how RPN avoids leaving poorly covered GT boxes without matched candidates.

## TV06: `RegionProposalNetwork.compute_loss`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/rpn.py`, lines 299-334](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/rpn.py#L299-L334)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Uses sampled-anchor BCE and positive-only Smooth L1 with beta 1/9.

**Why it matters:** Distinguishes actual default losses from modern IoU loss recipes.

## TV07: `RegionProposalNetwork.filter_proposals`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/rpn.py`, lines 242-297](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/rpn.py#L242-L297)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Performs per-level top-K, clipping, small-box/score filtering and level-grouped NMS.

**Why it matters:** Distinguishes proposal filtering from final class-wise NMS.

## TV08: `RoIHeads.select_training_samples`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/roi_heads.py`, lines 630-666](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/roi_heads.py#L630-L666)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Appends GT proposals, matches, samples and encodes regression targets.

**Why it matters:** Shows second assignment stage and why weak RPNs can still provide training positives.

## TV09: `fastrcnn_loss`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/roi_heads.py`, lines 12-49](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/roi_heads.py#L12-L49)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Uses softmax CE and class-specific positive-only Smooth L1 normalized by sampled RoI count.

**Why it matters:** Makes the stage-two background and regression conventions explicit.

## TV10: `RoIHeads.postprocess_detections`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/roi_heads.py`, lines 668-725](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/roi_heads.py#L668-L725)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Decodes class-specific boxes, softmaxes, removes background, thresholds and runs class-wise NMS.

**Why it matters:** Shows how many proposal-class candidates become final detections.

## TV11: `_resnet_fpn_extractor`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/backbone_utils.py`, lines 118-150](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/backbone_utils.py#L118-L150)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Selects residual stages and builds a 256-channel BackboneWithFPN.

**Why it matters:** Identifies the actual neck and per-model feature selections.

## TV12: `FeaturePyramidNetwork.forward`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/ops/feature_pyramid_network.py`, lines 172-204](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/ops/feature_pyramid_network.py#L172-L204)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Upsamples top-down inner features and adds lateral features before output convolution.

**Why it matters:** Separates additive FPN fusion from concatenation-based PAN designs.

## TV13: `FCOS.__init__`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/fcos.py`, lines 365-426](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/fcos.py#L365-L426)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Uses one stride-sized square reference per location and records inference defaults.

**Why it matters:** AnchorGenerator utility naming does not imply anchor-IoU-based assignment.

## TV14: `FCOS.compute_loss`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/fcos.py`, lines 440-487](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/fcos.py#L440-L487)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Selects positives by center sampling, inside-box and scale bounds, breaking ties by smallest GT area.

**Why it matters:** Provides center-sampling assignment evidence and explains many-to-one positives.

## TV15: `FCOSHead.compute_loss`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/fcos.py`, lines 52-125](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/fcos.py#L52-L125)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Uses sigmoid focal classification, decoded-box GIoU and positive centerness BCE, normalized by positives.

**Why it matters:** Documents the actual TorchVision loss recipe and its difference from generic FCOS summaries.

## TV16: `FCOSClassificationHead.forward`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/fcos.py`, lines 183-198](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/fcos.py#L183-L198)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Reshapes per-level dense class logits to N x HWA x C and concatenates levels.

**Why it matters:** Connects feature map locations to classifier targets and final candidate indexing.

## TV17: `FCOSRegressionHead.forward`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/fcos.py`, lines 245-267](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/fcos.py#L245-L267)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Applies ReLU to four regression outputs, predicts centerness from the regression tower, and flattens levels.

**Why it matters:** Establishes nonnegative distance prediction rather than anchor-delta regression.

## TV18: `BoxLinearCoder.encode`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/_utils.py`, lines 240-273](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/_utils.py#L240-L273)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Encodes ltrb distances from reference centers and normalizes by reference width/height.

**Why it matters:** Explains the stride-sized references used by anchor-free FCOS.

## TV19: `FCOS.postprocess_detections`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/fcos.py`, lines 489-556](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/fcos.py#L489-L556)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Computes square root of class-times-centerness sigmoid scores, per-level top-K, ltrb decoding and class-wise NMS.

**Why it matters:** Shows why FCOS uses NMS and why its confidence scale differs from softmax scores.

## TV20: `GeneralizedRCNNTransform.postprocess`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/transform.py`, lines 257-277](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/transform.py#L257-L277)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Resizes prediction boxes from transformed image size to original image size.

**Why it matters:** Prevents confusing resized/padded coordinates with delivered detections.

## TV21: `fcos_resnet50_fpn`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/fcos.py`, lines 766-775](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/fcos.py#L766-L775)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Builds ResNet50 FPN from layers 2,3,4 and P6/P7 extras.

**Why it matters:** Pins FCOS to P3-P7 rather than Faster R-CNN P2-P6.

## TV22: `resnet50`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/resnet.py`, lines 736-763](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/resnet.py#L736-L763)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Creates Bottleneck ResNet with stage block counts 3,4,6,3.

**Why it matters:** Identifies real backbone implementation rather than only a model-family label.

## TV23: `sigmoid_focal_loss`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/ops/focal_loss.py`, lines 7-58](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/ops/focal_loss.py#L7-L58)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Weights BCE by alpha_t and (1-p_t)^gamma with defaults alpha .25 and gamma 2.

**Why it matters:** Explains dense negative downweighting rather than negative subsampling.

## TV24: `Matcher.set_low_quality_matches_`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/_utils.py`, lines 398-417](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/_utils.py#L398-L417)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Restores candidates having the maximum available overlap for each GT, including ties.

**Why it matters:** Provides the best-anchor rescue mechanism invoked by RPN matching.

## TV25: `BoxLinearCoder.decode`

**Repository:** `pytorch/vision`  
**File:** [`torchvision/models/detection/_utils.py`, lines 275-311](https://github.com/pytorch/vision/blob/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/detection/_utils.py#L275-L311)  
**Pinned commit:** `3ac97aa9120137381ed1060f37237e44485ac2aa`

**What the code does:** Multiplies normalized ltrb by reference dimensions then subtracts/adds distances around reference centers.

**Why it matters:** Recovers xyxy detections from FCOS point/stride references without anchor-shape regression.

## Y01: `backbone / head graph`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/cfg/models/11/yolo11.yaml`, lines 15-47](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/cfg/models/11/yolo11.yaml#L15-L47)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Backbone, top-down and bottom-up fusion, and three Detect input levels are explicitly wired in YAML.

**Why it matters:** Distinguishes the fusion neck from the Detect prediction module.

## Y02: `parse_model`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/nn/tasks.py`, lines 933-1045](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/tasks.py#L933-L1045)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Parser applies compound scaling, rounds channels, injects repeated-block counts and Detect input-channel list.

**Why it matters:** Unscaled YAML channels are not the actual YOLO11n tensor channels.

## Y03: `C3k2.__init__`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/nn/modules/block.py`, lines 723-731](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/modules/block.py#L723-L731)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** C3k2 subclasses C2f and uses Bottleneck or C3k inner blocks.

**Why it matters:** Identifies actual backbone building block instead of assuming a ResNet backbone.

## Y04: `SPPF.forward`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/nn/modules/block.py`, lines 169-188](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/modules/block.py#L169-L188)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Sequential stride-one max pooling and concatenation preserve feature resolution.

**Why it matters:** Explains receptive-field expansion at P5.

## Y05: `C2PSA.forward`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/nn/modules/block.py`, lines 1037-1041](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/modules/block.py#L1037-L1041)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Projection splits channels, applies PSABlock to one part, and fuses it with bypass.

**Why it matters:** Attention is part of the backbone feature processing.

## Y06: `Detect`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/nn/modules/head.py`, lines 21-68](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/modules/head.py#L21-L68)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Standard Detect has end2end=False, reg_max=16, separated box/class branches and no objectness branch.

**Why it matters:** Defines both prediction representation and selected standard inference path.

## Y07: `Detect._inference`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/nn/modules/head.py`, lines 94-120](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/modules/head.py#L94-L120)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Flatten levels, integrate side distributions, decode around points, multiply by stride, and sigmoid class logits.

**Why it matters:** Connects dense raw predictions to resized-image boxes and class probabilities.

## Y08: `DFL`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/nn/modules/block.py`, lines 55-73](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/modules/block.py#L55-L73)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Fixed weights 0 through 15 calculate expected side distances after softmax.

**Why it matters:** Distributional regression must be decoded before box geometry is available.

## Y09: `DetectionModel.init_criterion`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/nn/tasks.py`, lines 386-388](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/tasks.py#L386-L388)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Non-end-to-end YOLO11 uses v8DetectionLoss.

**Why it matters:** Shared code names can describe newer model configurations without a separate loss class.

## Y10: `v8DetectionLoss`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/loss.py`, lines 157-260](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/loss.py#L157-L260)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** BCE classification; call-site TaskAligned topk10 alpha0.5 beta6; detached predictions for assignment; CIoU and DFL weighted losses.

**Why it matters:** Prevents mistaking commented varifocal code or unrelated focal classes for the active objective.

## Y11: `TaskAlignedAssigner`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/tal.py`, lines 90-125](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/tal.py#L90-L125)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Inside-box candidate masks and top-k alignment use s^alpha times clamped CIoU^beta.

**Why it matters:** Assignment uses both classification confidence and predicted geometry, rather than fixed anchor-box IoU.

## Y12: `TaskAlignedAssigner.forward`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/tal.py`, lines 72-88](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/tal.py#L72-L88)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Resolve collisions and normalize alignment scores to soft target scores.

**Why it matters:** Shows why classification labels carry localization quality.

## Y13: `TaskAlignedAssigner.select_highest_overlaps`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/tal.py`, lines 235-265](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/tal.py#L235-L265)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** When one point is assigned to multiple targets, keep the highest-overlap target.

**Why it matters:** Distinguishes conflict resolution from one-to-one set matching.

## Y14: `DFLoss.__call__`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/loss.py`, lines 73-88](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/loss.py#L73-L88)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Two-bin interpolated cross entropy supervises continuous side-distance targets.

**Why it matters:** DFL is a box-regression loss, not the classification focal loss.

## Y15: `make_anchors`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/tal.py`, lines 303-315](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/tal.py#L303-L315)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Generate center points with offset 0.5 and associate strides; no predefined aspect-ratio boxes.

**Why it matters:** Internal anchors naming does not make the detector anchor-box-based.

## Y16: `DetectionPredictor.postprocess`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/models/yolo/detect/predict.py`, lines 23-40](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/models/yolo/detect/predict.py#L23-L40)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Invoke confidence/IoU-configured NMS then scale boxes into the original image frame.

**Why it matters:** Traces the model-to-final-detections path.

## Y17: `non_max_suppression`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/ops.py`, lines 226-293](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/ops.py#L226-L293)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Confidence filter, xywh-to-xyxy conversion, optional class selection, pre-NMS truncation, class offsets, torchvision NMS and max_det.

**Why it matters:** Standard YOLO11 uses class-aware NMS rather than one-to-one query selection.

## Y18: `scale_boxes`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/ops.py`, lines 88-123](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/ops.py#L88-L123)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Subtract letterbox padding, divide by gain, and clip to original image dimensions.

**Why it matters:** Coordinates must be restored before comparing with original annotations.

## Y19: `BasePredictor.preprocess`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/engine/predictor.py`, lines 117-133](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/engine/predictor.py#L117-L133)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** BGR-to-RGB conversion, BCHW reordering and /255 normalization follow the pre_transform call.

**Why it matters:** Makes the preprocessing and box-restoration pair explicit.

## Y20: `BboxLoss.forward`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/loss.py`, lines 99-113](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/loss.py#L99-L113)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Quality-weighted CIoU and DFL supervise decoded boxes and side-distance distributions.

**Why it matters:** DFL is a box-regression loss, not the classification focal loss.

## Y21: `dist2bbox`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/utils/tal.py`, lines 318-327](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/utils/tal.py#L318-L327)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** Decode four ltrb distances into box coordinates relative to grid points.

**Why it matters:** Internal anchors naming does not make the detector anchor-box-based.

## Y22: `BasePredictor.pre_transform`

**Repository:** `ultralytics/ultralytics`  
**File:** [`ultralytics/engine/predictor.py`, lines 144-156](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/engine/predictor.py#L144-L156)  
**Pinned commit:** `6e43d1e1e5db72afbf686dee6745669bcb124b0a`

**What the code does:** LetterBox resizes images while preserving aspect ratio and adding padding.

**Why it matters:** Makes the preprocessing and box-restoration pair explicit.

## D01: `BackboneBase.__init__`

**Repository:** `facebookresearch/detr`  
**File:** [`models/backbone.py`, lines 58-70](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/backbone.py#L58-L70)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Detection mode returns only ResNet layer4; segmentation may request intermediate layers.

**Why it matters:** Original DETR is single-scale and has no FPN/PAN.

## D02: `Backbone.__init__`

**Repository:** `facebookresearch/detr`  
**File:** [`models/backbone.py`, lines 83-93](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/backbone.py#L83-L93)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** ResNet uses FrozenBatchNorm2d; ResNet-50 emits 2048 channels.

**Why it matters:** Identifies actual backbone rather than assigning a generic transformer-only backbone.

## D03: `DETR.__init__`

**Repository:** `facebookresearch/detr`  
**File:** [`models/detr.py`, lines 21-42](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L21-L42)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Creates class linear head, three-layer box MLP, learned object-query embedding and 1x1 feature projection.

**Why it matters:** Connects queries and output heads to concrete modules.

## D04: `DETR.forward`

**Repository:** `facebookresearch/detr`  
**File:** [`models/detr.py`, lines 59-72](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L59-L72)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Uses features[-1], transformer outputs, class head and sigmoid normalized box head.

**Why it matters:** Traces the actual final detection computation.

## D05: `Transformer.forward`

**Repository:** `facebookresearch/detr`  
**File:** [`models/transformer.py`, lines 47-59](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L47-L59)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Flattens spatial features, repeats learned query embeddings and initializes zero decoder content.

**Why it matters:** Explains how image features and object slots enter attention.

## D06: `TransformerDecoderLayer.forward_post`

**Repository:** `facebookresearch/detr`  
**File:** [`models/transformer.py`, lines 212-233](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L212-L233)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Runs query self-attention followed by cross-attention over encoder memory.

**Why it matters:** Query interaction is part of joint set prediction.

## D07: `HungarianMatcher.forward`

**Repository:** `facebookresearch/detr`  
**File:** [`models/matcher.py`, lines 34-82](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/matcher.py#L34-L82)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Combines negative class probability, pairwise L1 and negative GIoU, then invokes SciPy linear_sum_assignment.

**Why it matters:** One-to-one global assignment differs from dense many-positive training; matching cost is not cross entropy.

## D08: `SetCriterion.loss_labels`

**Repository:** `facebookresearch/detr`  
**File:** [`models/detr.py`, lines 108-127](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L108-L127)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Initializes every query to no-object, overwrites matched labels, and applies weighted cross entropy.

**Why it matters:** Unmatched duplicates receive no-object supervision.

## D09: `SetCriterion.loss_boxes`

**Repository:** `facebookresearch/detr`  
**File:** [`models/detr.py`, lines 143-162](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L143-L162)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Computes matched-box L1 and 1-GIoU normalized by object count.

**Why it matters:** Distinguishes differentiable box loss from matching cost.

## D10: `SetCriterion.forward`

**Repository:** `facebookresearch/detr`  
**File:** [`models/detr.py`, lines 215-253](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L215-L253)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Normalizes object count across workers and rematches every auxiliary decoder output.

**Why it matters:** Intermediate supervision does not reuse the final decoder assignment.

## D11: `PostProcess.forward`

**Repository:** `facebookresearch/detr`  
**File:** [`models/detr.py`, lines 258-286](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L258-L286)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Softmaxes logits, chooses foreground class per query, converts cxcywh to xyxy, and scales by target dimensions; no NMS, cutoff or top-K.

**Why it matters:** Verifies exact NMS-free inference and original-coordinate recovery.

## D12: `evaluate`

**Repository:** `facebookresearch/detr`  
**File:** [`engine.py`, lines 107-108](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/engine.py#L107-L108)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Passes each target orig_size to bbox postprocessor.

**Why it matters:** Confirms the rescaling dimensions are original image dimensions during evaluation.

## D13: `get_args_parser`

**Repository:** `facebookresearch/detr`  
**File:** [`main.py`, lines 35-79](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/main.py#L35-L79)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** CLI defaults are ResNet-50, sine encoding, 6+6 transformer layers, 256 dimensions, 8 heads, 100 queries, matching/loss coefficients 1:5:2 and eos_coef=0.1.

**Why it matters:** Actual build defaults override generic constructor defaults.

## D14: `build`

**Repository:** `facebookresearch/detr`  
**File:** [`models/detr.py`, lines 304-335](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L304-L335)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** COCO uses max category id plus one (91), so the classifier adds no-object for 92 outputs.

**Why it matters:** Avoids incorrectly reporting this original repository as an 81-logit contiguous COCO classifier.

## D15: `box_cxcywh_to_xyxy`

**Repository:** `facebookresearch/detr`  
**File:** [`util/box_ops.py`, lines 9-13](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/box_ops.py#L9-L13)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Unpacks cx,cy,w,h and constructs corner coordinates.

**Why it matters:** Actual box representation resolves a width/height typo in DETR.forward docstring.

## D16: `train_one_epoch`

**Repository:** `facebookresearch/detr`  
**File:** [`engine.py`, lines 32-35](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/engine.py#L32-L35)  
**Pinned commit:** `29901c51d7fe8712168b8d0d64351170bc0f83e0`

**What the code does:** Applies criterion.weight_dict to returned component losses.

**Why it matters:** Loss coefficients are applied outside SetCriterion.loss_boxes; cardinality logging is not optimized.
