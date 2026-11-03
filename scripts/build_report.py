#!/usr/bin/env python3
"""Build a ten-page, source-derived Track B report and editable Markdown."""
from pathlib import Path
import json
import re
from xml.sax.saxutils import escape

from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Frame, Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon
from reportlab.graphics import renderSVG

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor('#142A42')
TEAL = colors.HexColor('#047C88')
INK = colors.HexColor('#24384A')
LIGHT = colors.HexColor('#EFF5F8')
GRAY = colors.HexColor('#617181')
def _font_root():
    cands = [Path('/usr/share/fonts/truetype/dejavu')]
    try:
        import matplotlib
        cands.append(Path(matplotlib.__file__).parent / 'mpl-data' / 'fonts' / 'ttf')
    except ImportError:
        pass
    for c in cands:
        if (c / 'DejaVuSans.ttf').exists():
            return c
    raise FileNotFoundError('DejaVu fonts not found; install fonts-dejavu-core or matplotlib')
FONT_ROOT = _font_root()
for name, file in [('Body','DejaVuSans.ttf'), ('BodyBold','DejaVuSans-Bold.ttf'), ('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT/file)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
STYLES = {
 'body': ParagraphStyle('body',fontName='Body',fontSize=9.1,leading=13.2,textColor=INK,spaceAfter=7),
 'small': ParagraphStyle('small',fontName='Body',fontSize=7.8,leading=10.7,textColor=GRAY,spaceAfter=6),
 'h2': ParagraphStyle('h2',fontName='BodyBold',fontSize=11.3,leading=15,textColor=TEAL,spaceBefore=6,spaceAfter=7),
 'cell': ParagraphStyle('cell',fontName='Body',fontSize=7.6,leading=10.6,textColor=INK),
 'head': ParagraphStyle('head',fontName='BodyBold',fontSize=7.7,leading=10.6,textColor=colors.white),
 'code': ParagraphStyle('code',fontName='Mono',fontSize=7.4,leading=10.6,textColor=INK,spaceAfter=6,backColor=LIGHT,borderPadding=7),
}
MARKDOWN=[]

def p(text, style='body'):
    return Paragraph(text, STYLES[style])

def para(text, style='body'):
    MARKDOWN.append(re.sub('<[^>]+>','',text)+'\n')
    return p(text,style)

def h(text):
    MARKDOWN.append('### '+text+'\n')
    return p(text,'h2')

def tab(rows,widths):
    cells=[[p(escape(str(c)).replace('\n','<br/>'),'head' if i==0 else 'cell') for c in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
      ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
      ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
      ('LINEBELOW',(0,0),(-1,0),.6,TEAL),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#D9E3EA'))]))
    MARKDOWN.append('\n'+'\n'.join('| '+' | '.join(str(x).replace('\n',' / ') for x in row)+' |' for row in [rows[0],['---']*len(rows[0])]+rows[1:])+'\n')
    return t

def arrow(d,x1,y1,x2,y2):
    d.add(Line(x1,y1,x2,y2,strokeColor=TEAL,strokeWidth=1.1))
    if x2>x1: pts=[x2,y2,x2-5,y2+2.5,x2-5,y2-2.5]
    elif y2<y1: pts=[x2,y2,x2-2.5,y2+5,x2+2.5,y2+5]
    else: pts=[x2,y2,x2+5,y2+2.5,x2+5,y2-2.5]
    d.add(Polygon(pts,fillColor=TEAL,strokeColor=TEAL))

def diagram(name,boxes,caption,train):
    """Six nodes, serpentine order; explicit training/inference text stays outside."""
    d=Drawing(516,160)
    xs=[0,178,356,356,178,0]; ys=[111,111,111,49,49,49]
    d.add(Rect(0,0,516,30,rx=5,ry=5,fillColor=colors.white,strokeColor=TEAL,strokeWidth=.8,strokeDashArray=[3,2]))
    d.add(String(258,19,'TRAINING PATH (not run at inference)',fontName='BodyBold',fontSize=6.6,textAnchor='middle',fillColor=TEAL))
    d.add(String(258,8,train,fontName='Body',fontSize=6.9,textAnchor='middle',fillColor=INK))
    for i,(title,sub) in enumerate(boxes):
        x,y=xs[i],ys[i]
        d.add(Rect(x,y,160,43,rx=5,ry=5,fillColor=LIGHT if i<5 else colors.HexColor('#DCF0EF'),strokeColor=colors.HexColor('#C6D9E2'),strokeWidth=.7))
        d.add(String(x+80,y+26,title,fontName='BodyBold',fontSize=8.2,textAnchor='middle',fillColor=NAVY))
        d.add(String(x+80,y+11,sub,fontName='Body',fontSize=7.0,textAnchor='middle',fillColor=INK))
    arrow(d,160,132,178,132); arrow(d,338,132,356,132); arrow(d,436,111,436,92)
    arrow(d,356,70,338,70); arrow(d,178,70,160,70)
    out=ROOT/'docs'/'figures'/f'{name}.svg'; out.parent.mkdir(parents=True,exist_ok=True)
    renderSVG.drawToFile(d,str(out))
    MARKDOWN.append(f'![{caption}](figures/{name}.svg)\n')
    return [d,para(caption,'small')]

class Document:
    def __init__(self):
        out=ROOT/'report'/'2D_Object_Detection_Track_B.pdf'; out.parent.mkdir(exist_ok=True)
        self.c=canvas.Canvas(str(out),pagesize=(612,792))
        self.c.setTitle('2D Object Detection: A Source-Code Study of Four Detector Implementations')
        self.c.setAuthor('Aryan Dandwekar')
        self.n=0
    def page(self,title,subtitle,items):
        self.n+=1; c=self.c
        c.setFillColor(NAVY); c.rect(0,770,612,22,fill=1,stroke=0)
        c.setFont('BodyBold',8); c.setFillColor(colors.white); c.drawString(48,777,'INTELLIGENT AUTONOMOUS SYSTEMS  /  2D OBJECT DETECTION')
        c.setFillColor(NAVY); c.setFont('BodyBold',18); c.drawString(48,736,title)
        c.setFont('Body',8.6); c.setFillColor(GRAY); c.drawString(48,717,subtitle)
        c.setStrokeColor(TEAL); c.setLineWidth(1.5); c.line(48,706,564,706)
        c.setFillColor(GRAY); c.setFont('Body',7.4); c.drawString(48,28,'Aryan Dandwekar  |  Track B: code and architecture analysis  |  27 October 2026')
        c.drawRightString(564,28,f'{self.n} / 10')
        frame=Frame(48,49,516,646,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,showBoundary=0)
        remaining=list(items); frame.addFromList(remaining,c)
        if remaining: raise RuntimeError(f'Page {self.n} overflows: {len(remaining)} flowables remain')
        c.showPage()
    def finish(self):
        assert self.n==10
        self.c.save()
        (ROOT/'docs'/'report.md').write_text('# 2D Object Detection: A Source-Code Study\n\n'+'\n'.join(MARKDOWN))

def main():
    doc=Document()
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    commits={x['name']:x['commit'] for x in lock['repositories']}
    MARKDOWN.append('## 1. Scope, taxonomy and method\n')
    doc.page('Four detectors, traced through code','A technical implementation study of Faster R-CNN, FCOS, YOLO11 and DETR',[
      h('Research question and scope'),
      para('How do target assignment and output representation connect a detector\'s architecture, training losses and duplicate removal? This Track B investigation traces four real implementations across three families: two-stage region detectors, dense one-stage detectors, and query-based set predictors. It studies mechanisms rather than ranking published benchmark scores.'),
      para('The course <b>ngdet</b> tutorial [13] is the starting point: a detector must convert native outputs into consistent boxes, scores and labels before evaluation, which motivates inspecting those conversions. No tutorial accuracy or latency numbers are reported here.'),
      h('Detector evolution and taxonomy'),
      para('Detectors evolved along three axes. <b>Prediction unit:</b> region proposals (R-CNN family, Faster R-CNN [1] with FPN [2]) to dense grid predictions (SSD, YOLO, RetinaNet [4], FCOS [3]) to learned object queries (DETR [5]; Deformable DETR and RT-DETR follow but are not analyzed). <b>Reference geometry:</b> anchor boxes to anchor-free points. <b>Duplicate handling:</b> one-to-many assignment plus NMS to one-to-one Hungarian matching. YOLO11 reflects the dense branch after task-aligned assignment [6] and distribution-based box regression [7].'),
      tab([['Model / configuration','Paper','Source, framework, main class','Family'],['fasterrcnn_resnet50_fpn','[1,2]','TorchVision v0.20.1 [10], PyTorch; FasterRCNN','Two-stage, anchor-based'],['fcos_resnet50_fpn','[3,4]','TorchVision v0.20.1 [10], PyTorch; FCOS','Dense one-stage, anchor-free (geometric assignment)'],['YOLO11n (yolo11.yaml, scale n)','none; [6,7] for concepts','Ultralytics v8.3.0 [11], PyTorch; DetectionModel','Dense one-stage, anchor-free (task-aligned)'],['DETR ResNet-50, 100 queries','[5]','Official DETR [12], PyTorch; DETR','Query-based set predictor']],[118,82,172,144]),
      h('How the investigation was performed'),
      para('All four implementations use PyTorch. Official source trees were cloned, pinned by full commit SHA, and read through construction, loss and post-processing. Four original architecture diagrams were drawn from the inspected call paths. A source audit checks named definitions and file hashes; a separate CPU toy program illustrates assignment and NMS. These programs do not run detector inference or measure mAP.'),
      para('<b>Version discipline.</b> Historical versions are intentional: YOLO11 is the recent model in this selection, while the older implementations make architectural evolution inspectable. YOLO11 has no formal original research paper according to its official documentation [11]; its implementation is analyzed directly, with TOOD and GFL cited only for component concepts [6,7].'),
      para('Pinned commits: TorchVision '+commits['vision']+'; Ultralytics '+commits['ultralytics']+'; DETR '+commits['detr']+'. The companion repository (object-detection-code-study, submitted with this PDF) holds the full evidence index in docs/source_evidence.md.','small'),
    ])
    MARKDOWN.append('## 2. Faster R-CNN\n')
    doc.page('Faster R-CNN: two prediction stages','Exact configuration: torchvision.models.detection.fasterrcnn_resnet50_fpn [1,2,10]',[
      *diagram('faster_rcnn',[('Image transform','normalize, resize, batch padding'),('ResNet-50','resnet.py: Bottleneck stages'),('FPN: P2-P6','top-down + lateral features'),('RPN','objectness + anchor deltas'),('RoIAlign + two-MLP','class logits + class box deltas'),('Final detections','decode, filter, NMS, resize')],'Figure 1. Inference path derived from GeneralizedRCNN.forward, RPN.forward and RoIHeads.forward; the dashed strip shows training-time assignment and losses.','RPN: IoU matcher -> BCE + Smooth L1  |  RoIHeads: IoU 0.5 sampling -> softmax CE + Smooth L1'),
      h('Backbone, neck and heads'),
      para('<b>Construction.</b> faster_rcnn.py builds a ResNet-50 through _resnet_fpn_extractor in backbone_utils.py. BackboneWithFPN combines C2-C5 with FeaturePyramidNetwork; LastLevelMaxPool adds P6. RPN anchors have one scale per level (32, 64, 128, 256, 512) and three aspect ratios. MultiScaleRoIAlign uses P2-P5 to pool each proposal to 7 x 7. TwoMLPHead uses two 1,024-unit layers; FastRCNNPredictor emits class logits and class-specific box deltas.'),
      h('Training: two distinct matching problems'),
      para('<b>RPN.</b> RegionProposalNetwork.assign_targets_to_anchors uses box IoU and Matcher: IoU >= 0.7 is positive, &lt; 0.3 negative, intermediate matches ignored; low-quality matching also preserves best anchors for targets. A balanced sampler selects 256 anchors/image, at most half positive. compute_loss applies binary cross entropy to objectness and Smooth L1 (beta=1/9) to positive box deltas.'),
      para('<b>RoI head.</b> RoIHeads.select_training_samples appends ground-truth boxes, assigns proposals by IoU 0.5 and samples 512 proposals/image with positive fraction 0.25. fastrcnn_loss uses softmax cross entropy including background plus Smooth L1 on the positive class-specific deltas. Proposal generation detaches predicted deltas, so the discrete proposal-selection path is not differentiated through.'),
      h('Inference and implementation evidence'),
      para('RPN.filter_proposals performs per-level pre-NMS top-K, sigmoid scoring, clipping, filtering and level-aware NMS (0.7). RoIHeads.postprocess_detections decodes boxes, removes background, thresholds scores (0.05), removes tiny boxes, applies class-aware NMS (0.5) and keeps 100/image. GeneralizedRCNNTransform.postprocess rescales xyxy coordinates to the original image size. These are constructor defaults, not tuned experimental settings.'),
      para('<b>Code anchors</b> (torchvision/models/detection/): faster_rcnn.py: FasterRCNN.__init__; rpn.py: assign_targets_to_anchors, filter_proposals, compute_loss; roi_heads.py: select_training_samples, fastrcnn_loss, postprocess_detections; transform.py: GeneralizedRCNNTransform.postprocess; backbone_utils.py: _resnet_fpn_extractor.','small'),
    ])
    MARKDOWN.append('## 3. FCOS\n')
    doc.page('FCOS: points replace anchor templates','Exact configuration: torchvision.models.detection.fcos_resnet50_fpn [3,4,8,10]',[
      *diagram('fcos',[('Image transform','normalize, resize, pad'),('ResNet-50: C3-C5','resnet.py / BackboneWithFPN'),('FPN: P3-P7','LastLevelP6P7 adds coarse maps'),('FCOSHead','class + ltrb + centerness'),('Score, threshold, top-K','sqrt(cls x ctr); per-level 1000'),('Final detections','decode, clip, NMS, resize')],'Figure 2. Code-derived FCOS path. FCOS.compute_loss assigns targets; FCOSHead.compute_loss computes the three supervised objectives. Dashed strip: training path.','FCOS.compute_loss: center / scale / min-area -> sigmoid focal + GIoU + centerness BCE'),
      h('Why AnchorGenerator does not make FCOS anchor-based'),
      para('FCOS uses AnchorGenerator with one square per spatial location and side lengths 8, 16, 32, 64 and 128. In this implementation those squares encode a center and scale for BoxLinearCoder(normalize_by_size=True). Positives are not selected by overlap with a bank of aspect-ratio templates. The meaningful distinction is the matching and regression parameterization, rather than a helper class name.'),
      h('Architecture and geometric target assignment'),
      para('The factory returns C3-C5 through an FPN and LastLevelP6P7. FCOSClassificationHead and FCOSRegressionHead each use four convolutions with GroupNorm. The regression branch emits ReLU-constrained left, top, right and bottom distances; a parallel predictor emits centerness. Unlike a RoI head, every spatial location directly predicts a detection.'),
      para('FCOS.compute_loss restricts a positive point to: (1) a center region with radius 1.5 times the anchor-square side (equal to the level stride), (2) the inside of a ground-truth box, and (3) a level-dependent size interval. The first level has lower bound zero; the final level has upper bound infinity. If several boxes qualify, the smallest-area one wins. This deterministic geometric rule can assign many points to one object.'),
      h('Losses and prediction'),
      para('FCOSHead.compute_loss uses sigmoid focal classification loss, generalized_box_iou_loss on decoded positive boxes, and BCE-with-logits centerness loss, normalized by positive count. The centerness target is sqrt[min(l,r)/max(l,r) x min(t,b)/max(t,b)]. This implementation uses GIoU and stride-normalized ReLU distances; it should not be described simply as the original paper\'s unmodified regression recipe.'),
      para('postprocess_detections scores each class-point pair with sqrt(sigmoid(class) x sigmoid(centerness)), filters at 0.2, selects up to 1,000 pairs per level, decodes with BoxLinearCoder, clips boxes, applies class-aware NMS at 0.6, and retains 100/image. transform.postprocess restores original-image coordinates.'),
      para('<b>Code anchors</b> (torchvision/models/): detection/fcos.py: FCOS.compute_loss, FCOSHead.compute_loss, FCOS.postprocess_detections; detection/_utils.py: BoxLinearCoder; detection/backbone_utils.py: _resnet_fpn_extractor; ops/feature_pyramid_network.py: LastLevelP6P7.','small'),
    ])
    MARKDOWN.append('## 4. YOLO11\n')
    doc.page('YOLO11: task-aligned dense detection','Exact configuration: Ultralytics v8.3.0, yolo11.yaml, scale n [6,7,9,11]',[
      *diagram('yolo11',[('Image / LetterBox','aspect-ratio resize + padding'),('Conv + C3k2 backbone','SPPF and C2PSA at coarse stage'),('PAN-style fusion','upsample/concat; downsample/concat'),('Detect: P3, P4, P5','separate class / distribution paths'),('Decode + sigmoid','DFL expectation; anchor points'),('Final detections','confidence, class NMS, scale_boxes')],'Figure 3. yolo11.yaml layers 0-10 form the backbone; layers 11-22 fuse scales; layer 23 is Detect. Names come from code, not an external diagram. Dashed strip: training path.','v8DetectionLoss: TaskAlignedAssigner (top-10) -> BCE on soft targets + CIoU + DFL'),
      h('Construction and output representation'),
      para('DetectionModel and parse_model in nn/tasks.py resolve the YAML into modules and apply depth/width scaling. C3k2 implements CSP-style feature processing; SPPF aggregates pooled context; C2PSA adds attention at the coarse backbone stage. The YAML head performs both top-down and bottom-up feature fusion before Detect consumes layers 16, 19 and 22.'),
      para('For a theoretical square 640 x 640 input, YOLO11n has head inputs B x 64 x 80 x 80, B x 128 x 40 x 40 and B x 256 x 20 x 20. Detect produces 80 class logits and 4 x 16 distance-bin logits per point: 144 channels, across 8,400 points. DFL integrates each distribution to a distance; decoded inference output is B x 84 x 8400. These shapes are statically derived, not a recorded model forward pass.'),
      h('Assignment and actual training losses'),
      para('DetectionModel.init_criterion selects v8DetectionLoss despite the model being YOLO11. That criterion calls TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0) on detached predictions. Eligible points must lie inside targets; alignment combines class probability and a localization term. This pinned implementation uses clamped CIoU for that term, resolves shared candidates by overlap, and creates normalized soft class targets.'),
      para('The actual objective is BCE-with-logits against those targets, weighted CIoU box loss and Distribution Focal Loss. It is not standard focal loss or VarifocalLoss merely because those classes also exist in loss.py. For continuous distance y between bins l=floor(y) and r=l+1, DFL interpolates two cross entropies: (r-y)CE(l)+(y-l)CE(r). The inference DFL module computes an expectation; it is not the training loss.'),
      para('<b>Code anchors</b> (ultralytics/): cfg/models/11/yolo11.yaml; nn/tasks.py: DetectionModel, parse_model; nn/modules/block.py: C3k2, SPPF, C2PSA; nn/modules/head.py: Detect; utils/loss.py: v8DetectionLoss; utils/tal.py: TaskAlignedAssigner.','small'),
      h('Inference: follow the selected branch'),
      para('Detect.end2end is False here. Detect._inference decodes point distances and applies class sigmoid, then DetectionPredictor.postprocess calls non_max_suppression (default confidence 0.25, IoU 0.7, max_det 300). scale_boxes removes letterbox padding, divides by gain and clips to the original image. The shared file\'s end-to-end/top-K branch exists for other models and is not the standard YOLO11 route.'),
    ])
    MARKDOWN.append('## 5. DETR\n')
    doc.page('DETR: learn a set of detections','Exact configuration: official DETR, ResNet-50, 100 learned queries [5,8,12]',[
      *diagram('detr',[('Image + padding mask','NestedTensor and positional codes'),('ResNet-50: final C5','1 x 1 projection to 256 channels'),('Transformer encoder','flatten spatial features + positions'),('Transformer decoder','100 learned query embeddings'),('Parallel prediction heads','class incl. no-object; sigmoid boxes'),('Final detections','class max, xyxy, original-size scaling')],'Figure 4. DETR.forward uses features[-1]; this baseline does not add FPN/PAN or fuse a multi-scale pyramid. Hungarian matching supports supervised loss, not prediction post-processing. Dashed strip: training path.','SetCriterion: HungarianMatcher -> CE (no-object weight 0.1) + L1 + GIoU, plus auxiliary decoder losses'),
      h('Backbone, attention and heads'),
      para('models/backbone.py uses a TorchVision ResNet with FrozenBatchNorm2d and an IntermediateLayerGetter. DETR.forward selects the last feature map and projects it with input_proj. transformer.py flattens that map to a token sequence. The standard main.py arguments specify 256 hidden dimensions, eight attention heads and six encoder/six decoder layers. Learned query embeddings condition the decoder; its initial target content is zeros.'),
      para('class_embed maps each decoder state to K+1 logits, including no-object; bbox_embed is a three-layer MLP producing four sigmoid-normalized cx,cy,w,h values. K is the configured label-index space: the official COCO builder uses num_classes=91, so the output has 92 logits, rather than assuming exactly 80 contiguous classes plus background. This is a practical taxonomy detail for adapters.'),
      h('Matching cost is not the differentiable loss'),
      para('HungarianMatcher.forward constructs a rectangular query-target cost: -class probability + weighted L1 distance - weighted GIoU, using weights 1, 5 and 2 in the default configuration. It transfers costs to CPU and calls scipy.optimize.linear_sum_assignment. This discrete assignment runs under no_grad and selects at most one query per target.'),
      para('SetCriterion.loss_labels applies cross entropy to every query, assigning unmatched queries the no-object class with eos_coef=0.1. loss_boxes applies L1 and 1-GIoU only to matched boxes. Optional auxiliary decoder outputs are rematched and supervised too. Thus -p in matching and -log(p) in classification training are distinct operations; the source explicitly separates them.'),
      h('Inference: no assignment and no NMS'),
      para('PostProcess.forward takes softmax over all K+1 logits, excludes the final no-object column when taking the best foreground class, converts cxcywh to xyxy and multiplies by [W,H,W,H] from target_sizes. It returns one foreground-scored candidate per query. It does not call Hungarian matching, NMS, a score threshold or top-K. A visualization/client may filter afterward; those choices are outside this PostProcess implementation.'),
      para('<b>Code anchors</b> (detr/models/): backbone.py: Backbone, Joiner; transformer.py: Transformer; detr.py: DETR.forward, SetCriterion, PostProcess; matcher.py: HungarianMatcher.','small'),
    ])
    MARKDOWN.append('## 6. Target assignment and loss comparison\n')
    doc.page('Target assignment changes supervision','Four strategies compared at their actual training call sites',[
      para('A target assigner decides which predictions receive a foreground label and regression target. This choice changes the number of positives, the competing negative gradients and whether several outputs are trained to represent the same object. It is therefore a core difference among families, not a minor preprocessing detail.'),
      tab([['Strategy / call site','Positive rule and ambiguity','Consequence'],['Anchor/IoU matching\nRPN.assign_targets_to_anchors','Match geometric templates by IoU; ignore a band; rescue best anchors. RoI stage rematches proposals.','One object can supervise many anchors/proposals; sampling controls imbalance.'],['Center sampling\nFCOS.compute_loss','Point inside box and center region, compatible feature scale; smallest area breaks ties.','Removes aspect-ratio templates, but retains spatial/scale priors and many positives.'],['Task-aligned assignment\nTaskAlignedAssigner.forward','Inside-target candidates; top-10 by p^0.5 x u^6, u = max(CIoU,0); overlap resolves conflicts.','Predicted classification and localization jointly determine soft targets; one-to-many.'],['Hungarian matching\nHungarianMatcher.forward','Globally minimize query-target cost with a unique query for each matched target.','One-to-one foreground supervision; unmatched queries learn no-object.']],[142,222,152]),
      h('Classification imbalance and box geometry'),
      para('<b>Focal loss [4].</b> FL(p_t)=-alpha_t(1-p_t)^gamma log(p_t) reduces easy-example weight; FCOS uses sigmoid focal loss (alpha=0.25, gamma=2). Faster R-CNN instead samples training examples and uses BCE/CE. YOLO11 uses quality-weighted soft targets with BCE, while DETR weights its no-object CE. DFL [7] concerns distance distributions and should not be confused with focal classification.'),
      para('<b>Box losses [8,9].</b> IoU measures intersection divided by union. GIoU also penalizes unused area in the smallest enclosing rectangle, giving a signal for disjoint boxes; this appears in FCOS and DETR. CIoU adds center-distance and aspect-ratio penalties and appears in YOLO11. Faster R-CNN uses Smooth L1 on encoded deltas. Their raw loss magnitudes are not directly comparable.'),
      h('A reproducible toy check, separate from model execution'),
      para('scripts/toy_assignment_demo.py uses explicit boxes and a tiny cost matrix to demonstrate several dense positives for one target, greedy NMS suppression, and globally optimal one-to-one assignment by exhaustive enumeration. Enumerating the tiny problem verifies the same optimization objective used by Hungarian matching; it is not an implementation or timing benchmark of the Hungarian algorithm. The toy run produced three dense positive pairs but two one-to-one matches; NMS kept predictions 0 and 2. A crowded-object case also shows NMS suppressing a distinct same-class object. These are synthetic outputs, not detector results.'),
    ])
    MARKDOWN.append('## 7. Code-level comparison and coordinate recovery\n')
    doc.page('Implementation comparison','All properties refer to the pinned configurations, not every model in a family',[
      tab([['Property','Faster R-CNN','FCOS','YOLO11n','DETR-R50'],['Backbone','ResNet-50','ResNet-50','Conv, C3k2, SPPF, C2PSA','ResNet-50'],['Neck / feature path','FPN P2-P6; RoI uses P2-P5','FPN P3-P7','PAN-style P3-P5 fusion','C5 projection + transformer; no FPN'],['Anchor templates?','Yes, 3 ratios per level','No; stride squares are coding helpers','No; anchor points','No'],['Target assignment','IoU twice + sampling','Center / scale / min-area','Task-aligned, top-10','Hungarian one-to-one'],['Classification','RPN BCE; RoI softmax CE','Sigmoid focal loss','BCE on soft targets','Softmax CE + no-object weight'],['Box loss','Smooth L1 deltas','GIoU decoded boxes','CIoU + DFL','L1 + GIoU'],['Other supervision','RPN objectness','Centerness BCE','No separate objectness output','Auxiliary decoder losses optional'],['Object queries?','No','No','No','100 learned slots'],['NMS?','RPN per-level + final per-class','Per-class','Per-class, standard path','None in PostProcess'],['Selection budget','RPN top-K; final 100','1,000 per level; final 100','NMS candidate cap; final 300','100 queries; no post top-K'],['Main class / file (TorchVision: models/detection/)','FasterRCNN\nfaster_rcnn.py','FCOS\nfcos.py','DetectionModel\nnn/tasks.py','DETR\nmodels/detr.py']],[92,106,106,106,106]),
      h('From network coordinates to original-image boxes'),
      para('<b>TorchVision:</b> GeneralizedRCNNTransform records original and resized dimensions, decodes/clips boxes in resized-image coordinates and scales x/y separately in postprocess. <b>Ultralytics:</b> LetterBox adds padding; ops.scale_boxes reverses padding and the resize gain before clipping. <b>DETR:</b> PostProcess scales normalized xyxy by each original W,H. It does not undo an arbitrary letterbox transform; an adapter must respect the preprocessing contract.'),
      para('A common score threshold across families can be a reproducible operating point, but it does not guarantee comparable recall or calibration. Full AP evaluation should retain sufficiently low-scoring candidates and document truncation/top-K. These are design implications from the code, not measured performance findings.','small'),
    ])
    MARKDOWN.append('## 8. Required deep dive: YOLO to DETR\n')
    doc.page('Why YOLO11 uses NMS and DETR can omit it','Deep dive: connect the training assignment to the actual inference branch',[
      h('1. Dense supervision intentionally rewards several predictions'),
      para('In YOLO11\'s v8DetectionLoss.__call__, decoded detached boxes and class probabilities are passed to TaskAlignedAssigner. select_topk_candidates can nominate up to ten eligible points for one ground-truth box. After conflict resolution, several points can still represent that same object. BCE, CIoU and DFL then reward those multiple positive predictions. The loss does not require exactly one surviving prediction for that object.'),
      h('2. NMS converts overlapping candidates into detections'),
      para('Detect._inference only decodes distributions and class probabilities. DetectionPredictor.postprocess invokes ops.non_max_suppression; its class-offset mechanism makes default suppression class-aware. Greedy NMS keeps the best-scoring box, suppresses sufficiently overlapping boxes of that class, then repeats. This provides a selection rule after one-to-many training. NMS can also suppress a correct nearby same-class object in crowded scenes; that is a mechanism-based failure hypothesis, not an observed error from this work.'),
      h('3. DETR assigns object identity during training'),
      para('HungarianMatcher.forward instead selects an injective assignment between targets and queries. SetCriterion.loss_labels fills every unmatched slot with the no-object label before cross entropy. Matching chooses a foreground representative, and duplicate unmatched predictions receive pressure toward no-object. Decoder self-attention also allows queries to interact. The combination supports learning a set whose members represent different objects [5].'),
      h('4. NMS-free is a learned behavior, not a proof of uniqueness'),
      para('At inference, PostProcess.forward performs a foreground-class maximum and coordinate conversion only. No ground truth or Hungarian solver is used. Two queries can still produce similar boxes, and an unfiltered query always has some best foreground score even when no-object is most probable. NMS-free describes the intended trained model and its pipeline; it is not a hard geometric constraint that makes duplicates impossible.'),
      h('5. The distinction is not CNN versus transformer'),
      para('YOLO11\'s shared head.py contains forward_end2end and a top-K postprocess method, but Detect.end2end=False routes this specific model through the dense path. Reading a nearby unused function would lead to the wrong conclusion. Conversely, merely deleting NMS from standard YOLO11 would leave its one-to-many training unchanged. A convincing NMS-free design needs compatible assignment, objectives and output selection, regardless of backbone type.'),
      para('<b>Evidence chain:</b> Ultralytics utils/loss.py: v8DetectionLoss.__call__ -> utils/tal.py: TaskAlignedAssigner -> nn/modules/head.py: Detect._inference -> models/yolo/detect/predict.py: DetectionPredictor.postprocess -> utils/ops.py: non_max_suppression. Contrast DETR models/matcher.py: HungarianMatcher.forward -> models/detr.py: SetCriterion.loss_labels and PostProcess.forward. Each link is pinned and audited in the companion repository.','small'),
    ])
    MARKDOWN.append('## 9. Evidence, reproducibility and conclusions\n')
    doc.page('Evidence and reproducibility','The full evidence index contains paths, symbols, line spans, reasons and immutable links',[
      tab([['Repository-relative path / lines','Inspected symbol','What it does / why it matters'],['[10] torchvision/models/detection/rpn.py:193-229','RegionProposalNetwork.\nassign_targets_to_anchors','IoU matching and best-anchor rescue determine positives.'],['[10] torchvision/models/detection/fcos.py:440-487','FCOS.compute_loss','Center/scale rules, rather than anchor IoU, select targets.'],['[10] torchvision/models/detection/fcos.py:52-125','FCOSHead.compute_loss','Actual objectives: focal, GIoU and centerness BCE.'],['[11] ultralytics/nn/modules/head.py:21-68','Detect.forward','end2end=False selects standard dense prediction.'],['[11] ultralytics/utils/loss.py:207-260','v8DetectionLoss.__call__','Active classification loss is BCE, unlike unused alternatives.'],['[11] ultralytics/utils/tal.py:102-125','TaskAlignedAssigner.\nget_box_metrics','Class score and clamped CIoU define alignment quality.'],['[12] models/matcher.py:34-82','HungarianMatcher.forward','Global one-to-one assignment for supervised loss.'],['[12] models/detr.py:258-286','PostProcess.forward','Foreground score selection and scaling; no NMS.'],['[12] models/detr.py:108-127','SetCriterion.loss_labels','Unmatched queries get no-object supervision, pressuring duplicates down.'],['[10] torchvision/models/detection/fcos.py:489-556','FCOS.postprocess_detections','sqrt(cls x centerness) scoring, per-level top-K, class-wise NMS.']],[193,145,178]),
      h('Reproduce the investigation'),
      para('From the repository root (network needed only to fetch the pinned trees; PDF building needs requirements.txt):'),
      para('python scripts/fetch_sources.py<br/>python scripts/inspect_sources.py<br/>python scripts/toy_assignment_demo.py<br/>python scripts/build_report.py','code'),
      para('sources.lock.json pins commits; docs/evidence.json holds 63 source locations; artifacts/ holds the audit hashes and toy outputs; docs/model_notes/ has deeper call-path notes. No weights or datasets are bundled.'),
      h('Conclusions, limits and AI use'),
      para('The clearest family difference is the contract between assignment and prediction: sampled anchor/proposal matching, geometric point matching, prediction-aligned dense matching, or global one-to-one matching. Those contracts explain the loss terms and duplicate handling more directly than model names. Static source analysis establishes intended implementation behavior; it cannot establish accuracy, runtime, memory use or robustness.'),
      para('AI tools assisted source navigation, explanations, scripts, diagrams and report drafting. Claims were checked against the pinned code and original papers; the included source audit verifies source locations, not the truth of every interpretation. No detector training or inference was run. The diagrams and toy examples were created for this investigation.','small'),
    ])
    MARKDOWN.append('## 10. References\n')
    refs=[
      ('[1] S. Ren, K. He, R. Girshick and J. Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS, 2015.','https://arxiv.org/abs/1506.01497'),
      ('[2] T.-Y. Lin et al. Feature Pyramid Networks for Object Detection. CVPR, 2017.','https://arxiv.org/abs/1612.03144'),
      ('[3] Z. Tian, C. Shen, H. Chen and T. He. FCOS: Fully Convolutional One-Stage Object Detection. ICCV, 2019.','https://arxiv.org/abs/1904.01355'),
      ('[4] T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollar. Focal Loss for Dense Object Detection. ICCV, 2017.','https://arxiv.org/abs/1708.02002'),
      ('[5] N. Carion et al. End-to-End Object Detection with Transformers. ECCV, 2020.','https://arxiv.org/abs/2005.12872'),
      ('[6] C. Feng, Y. Zhong, Y. Gao, M. R. Scott and W. Huang. TOOD: Task-aligned One-stage Object Detection. ICCV, 2021.','https://arxiv.org/abs/2108.07755'),
      ('[7] X. Li et al. Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection. NeurIPS, 2020.','https://arxiv.org/abs/2006.04388'),
      ('[8] H. Rezatofighi et al. Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression. CVPR, 2019.','https://arxiv.org/abs/1902.09630'),
      ('[9] Z. Zheng et al. Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. AAAI, 2020. Introduces CIoU as well as DIoU.','https://arxiv.org/abs/1911.08287'),
      ('[10] PyTorch. TorchVision source, v0.20.1. Established framework implementation of Faster R-CNN and FCOS; this is not a claim to inspect the original FCOS author code.','https://github.com/pytorch/vision/tree/'+commits['vision']),
      ('[11] Ultralytics. YOLO11 source, v8.3.0, and official YOLO11 documentation. A formal original YOLO11 paper is not available.','https://github.com/ultralytics/ultralytics/tree/'+commits['ultralytics']),
      ('[12] Meta / Facebook Research. Official DETR author implementation.','https://github.com/facebookresearch/detr/tree/'+commits['detr']),
      ('[13] K. K. Liu. ngdet Tutorial: Zero-shot 2D detection across datasets. Course-provided TUTORIAL.md; uploaded snapshot used as the starting point.','https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md'),
    ]
    items=[para('Nine original research papers and three official source repositories support this analysis. Code links use immutable commits; the course tutorial and live documentation are identified separately. Access date: 27 October 2026.','small')]
    for text,url in refs:
        items.append(para(escape(text)+f'<br/><link href="{escape(url)}" color="#047C88">{escape(url)}</link>','small'))
    items.append(para('Additional official documentation for [11]: <link href="https://docs.ultralytics.com/models/yolo11/" color="#047C88">https://docs.ultralytics.com/models/yolo11/</link>. The README links the editable report, original diagrams and full source-code evidence index.','small'))
    doc.page('References','Original papers, official implementations and the course starting point',items)
    doc.finish()
    print('Built report/2D_Object_Detection_Track_B.pdf (10 pages) and docs/report.md')

if __name__=='__main__': main()
