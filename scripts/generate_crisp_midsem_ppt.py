"""
MedBoard — scripts/generate_crisp_midsem_ppt.py
Generates the crisp, 5-section Mid-Semester Evaluation Presentation (.pptx)
strictly matching the NIT Kurukshetra notification guidelines without decomposition:

  1. Problem Identification and Objectives
  2. Literature Review
  3. Methodology and Work Plan
  4. Progress Achieved
  5. Future Work and Timeline

100% realistic, grounded in actual repository results, professional static formatting.
"""

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Static Professional Academic Palette ──────────────────────────────────────
PRIMARY_NAVY    = RGBColor(26, 54, 93)      # #1A365D - Formal Deep Navy
SECONDARY_NAVY  = RGBColor(43, 108, 176)    # #2B6CB0 - Steel Blue Accent
TEXT_DARK       = RGBColor(45, 55, 72)      # #2D3748 - Charcoal Text
TEXT_MUTED      = RGBColor(113, 128, 150)   # #718096 - Slate Gray
BG_CARD         = RGBColor(248, 250, 252)   # #F8FAFC - Clean Card Fill
BORDER_CARD     = RGBColor(203, 213, 225)   # #CBD5E0 - Subtle 1pt Border
BORDER_HEADER   = RGBColor(226, 232, 240)   # #E2E8F0 - Divider Line
WHITE           = RGBColor(255, 255, 255)
HIGHLIGHT_ROW   = RGBColor(237, 242, 247)   # #EDF2F7 - Selected Table Row

SLIDE_WIDTH  = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)


def add_slide_header(slide, section_number: str, section_title: str, slide_num: int):
    """Adds a standard, static academic header across all content slides."""
    # Top eyebrow tracker
    tb_eye = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.73), Inches(0.28))
    tf_eye = tb_eye.text_frame
    tf_eye.word_wrap = True
    tf_eye.margin_left = tf_eye.margin_top = tf_eye.margin_right = tf_eye.margin_bottom = 0
    p_eye = tf_eye.paragraphs[0]
    p_eye.text = f"MID-SEMESTER EVALUATION • SECTION {section_number}"
    p_eye.font.name = "Calibri"
    p_eye.font.size = Pt(9.5)
    p_eye.font.bold = True
    p_eye.font.color.rgb = SECONDARY_NAVY

    # Section Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.73), Inches(0.55))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_t = tf_title.paragraphs[0]
    p_t.text = section_title
    p_t.font.name = "Calibri"
    p_t.font.size = Pt(21)
    p_t.font.bold = True
    p_t.font.color.rgb = PRIMARY_NAVY

    # Thin divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.73), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_HEADER
    line.line.fill.background()

    # Footer note & slide number
    tb_foot = slide.shapes.add_textbox(Inches(0.8), Inches(7.1), Inches(10.5), Inches(0.3))
    tf_foot = tb_foot.text_frame
    tf_foot.margin_left = tf_foot.margin_top = tf_foot.margin_right = tf_foot.margin_bottom = 0
    p_foot = tf_foot.paragraphs[0]
    p_foot.text = "NIT Kurukshetra • Department of Computer Engineering • B.Tech Project (CSPC401 & ITPC401)"
    p_foot.font.name = "Calibri"
    p_foot.font.size = Pt(8.5)
    p_foot.font.color.rgb = TEXT_MUTED

    tb_num = slide.shapes.add_textbox(Inches(11.53), Inches(7.1), Inches(1.0), Inches(0.3))
    tf_num = tb_num.text_frame
    tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
    p_num = tf_num.paragraphs[0]
    p_num.text = f"{slide_num} / 7"
    p_num.font.name = "Calibri"
    p_num.font.size = Pt(8.5)
    p_num.font.bold = True
    p_num.font.color.rgb = TEXT_MUTED
    p_num.alignment = PP_ALIGN.RIGHT


def add_card(slide, left, top, width, height, bg_color=BG_CARD, border_color=BORDER_CARD):
    """Draws a rounded card container with crisp border."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.0)
    else:
        shape.line.fill.background()
    return shape


def add_bullet_point(text_frame, bold_prefix: str, content: str, pt_size: float = 10.5, space_after: float = 6.0):
    """Adds a standard bullet point with bold lead."""
    p = text_frame.add_paragraph() if len(text_frame.paragraphs[0].text) > 0 else text_frame.paragraphs[0]
    p.space_after = Pt(space_after)
    p.font.name = "Calibri"
    p.font.size = Pt(pt_size)

    r_bold = p.add_run()
    r_bold.text = bold_prefix + (" " if bold_prefix else "")
    r_bold.font.bold = True
    r_bold.font.color.rgb = PRIMARY_NAVY

    r_text = p.add_run()
    r_text.text = content
    r_text.font.bold = False
    r_text.font.color.rgb = TEXT_DARK


def build_presentation(output_path: str = "docs/MedBoard_MidSem_Official.pptx"):
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # Top Banner
    top_rect = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_WIDTH, Inches(3.2))
    top_rect.fill.solid()
    top_rect.fill.fore_color.rgb = PRIMARY_NAVY
    top_rect.line.fill.background()

    # Institute Info
    tb_inst = s1.shapes.add_textbox(Inches(0.9), Inches(0.4), Inches(11.5), Inches(0.7))
    tf_inst = tb_inst.text_frame
    tf_inst.word_wrap = True
    p1 = tf_inst.paragraphs[0]
    p1.text = "NATIONAL INSTITUTE OF TECHNOLOGY KURUKSHETRA"
    p1.font.name = "Calibri"
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(226, 232, 240)

    p2 = tf_inst.add_paragraph()
    p2.text = "DEPARTMENT OF COMPUTER ENGINEERING • AY: 2026-27 (ODD SEMESTER)"
    p2.font.name = "Calibri"
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(2)

    # Project Title
    tb_t = s1.shapes.add_textbox(Inches(0.9), Inches(1.25), Inches(11.5), Inches(1.7))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    pt = tf_t.paragraphs[0]
    pt.text = "MedBoard: Brain MRI Segmentation and Classification System"
    pt.font.name = "Calibri"
    pt.font.size = Pt(25)
    pt.font.bold = True
    pt.font.color.rgb = WHITE
    pt.space_after = Pt(4)

    pt_sub = tf_t.add_paragraph()
    pt_sub.text = "Mid-Semester Project Evaluation • Course: CSPC401 & ITPC401"
    pt_sub.font.name = "Calibri"
    pt_sub.font.size = Pt(13)
    pt_sub.font.color.rgb = RGBColor(191, 219, 254)

    # Bottom Half: Metadata Cards
    # Left Card: Team Members
    add_card(s1, Inches(0.9), Inches(3.55), Inches(5.6), Inches(3.25), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_m = s1.shapes.add_textbox(Inches(1.15), Inches(3.7), Inches(5.1), Inches(2.95))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True
    pm_t = tf_m.paragraphs[0]
    pm_t.text = "PROJECT TEAM MEMBERS"
    pm_t.font.name = "Calibri"
    pm_t.font.size = Pt(11)
    pm_t.font.bold = True
    pm_t.font.color.rgb = PRIMARY_NAVY
    pm_t.space_after = Pt(8)

    members = [
        ("Pragya Thakur", "123102162", "B.Tech Computer Engineering"),
        ("Shivanshu Sajwan", "123102193", "B.Tech Computer Engineering"),
        ("Digvijay Singh", "123102198", "B.Tech Computer Engineering"),
    ]
    for name, roll, dept in members:
        p_mem = tf_m.add_paragraph()
        p_mem.space_after = Pt(6)
        r_n = p_mem.add_run()
        r_n.text = f"•  {name} "
        r_n.font.bold = True
        r_n.font.size = Pt(11.5)
        r_n.font.color.rgb = PRIMARY_NAVY
        r_r = p_mem.add_run()
        r_r.text = f"({roll})\n   {dept}"
        r_r.font.size = Pt(9.5)
        r_r.font.color.rgb = TEXT_MUTED

    # Right Card: Coordinator & Committee
    add_card(s1, Inches(6.8), Inches(3.55), Inches(5.6), Inches(3.25), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_c = s1.shapes.add_textbox(Inches(7.05), Inches(3.7), Inches(5.1), Inches(2.95))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    pc_t = tf_c.paragraphs[0]
    pc_t.text = "FACULTY COORDINATOR & EVALUATION COMMITTEE"
    pc_t.font.name = "Calibri"
    pc_t.font.size = Pt(11)
    pc_t.font.bold = True
    pc_t.font.color.rgb = PRIMARY_NAVY
    pc_t.space_after = Pt(8)

    add_bullet_point(tf_c, "Project Coordinator / Mentor:", "Dr. Vishwas Rathi", pt_size=11.0, space_after=8.0)
    add_bullet_point(tf_c, "Evaluation Committee Members:", "Dr. Niyati Baliyan\nDr. Vijay Kumar Sharma", pt_size=11.0, space_after=8.0)
    add_bullet_point(tf_c, "Project Type & Scope:", "B.Tech Major Capstone Project (CSPC401 / ITPC401)", pt_size=10.0, space_after=0.0)

    # =========================================================================
    # SLIDE 2 (SECTION 1): PROBLEM IDENTIFICATION AND OBJECTIVES
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "1", "Problem Identification and Objectives", slide_num=2)

    # Left Column: Problem Identification
    add_card(s2, Inches(0.8), Inches(1.55), Inches(5.7), Inches(5.35), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_prob = s2.shapes.add_textbox(Inches(1.05), Inches(1.7), Inches(5.2), Inches(5.05))
    tf_prob = tb_prob.text_frame
    tf_prob.word_wrap = True

    p_phead = tf_prob.paragraphs[0]
    p_phead.text = "PROBLEM IDENTIFICATION"
    p_phead.font.name = "Calibri"
    p_phead.font.size = Pt(12.5)
    p_phead.font.bold = True
    p_phead.font.color.rgb = PRIMARY_NAVY
    p_phead.space_after = Pt(10)

    add_bullet_point(tf_prob, "Clinical Triage Burden:",
                     "Manual evaluation of brain MRI slices is time-consuming and subject to inter-observer variability, compounded by a global shortage of specialized neuroradiologists.",
                     pt_size=10.5, space_after=10.0)
    add_bullet_point(tf_prob, "Dual Diagnostic Challenge:",
                     "Clinicians require both accurate tumor type identification (Glioma, Meningioma, Pituitary, Normal) and precise anatomical boundary delineation (segmentation) for biopsy and surgical planning.",
                     pt_size=10.5, space_after=10.0)
    add_bullet_point(tf_prob, "Infiltrative & Ambiguous Margins:",
                     "Intracranial neoplasms exhibit irregular, low-contrast boundaries against normal parenchyma, causing standard thresholding and shallow architectures to under-segment or leak into ventricles.",
                     pt_size=10.5, space_after=10.0)
    add_bullet_point(tf_prob, "Fragmented Tooling:",
                     "Existing computer-aided diagnostic systems typically handle classification or segmentation in isolation, lacking a unified pipeline and interactive clinical interface.",
                     pt_size=10.5, space_after=0.0)

    # Right Column: Objectives
    add_card(s2, Inches(6.83), Inches(1.55), Inches(5.7), Inches(5.35), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_obj = s2.shapes.add_textbox(Inches(7.08), Inches(1.7), Inches(5.2), Inches(5.05))
    tf_obj = tb_obj.text_frame
    tf_obj.word_wrap = True

    p_ohead = tf_obj.paragraphs[0]
    p_ohead.text = "PROJECT OBJECTIVES"
    p_ohead.font.name = "Calibri"
    p_ohead.font.size = Pt(12.5)
    p_ohead.font.bold = True
    p_ohead.font.color.rgb = PRIMARY_NAVY
    p_ohead.space_after = Pt(10)

    add_bullet_point(tf_obj, "1. Automated 4-Class Classification:",
                     "Train a lightweight, high-accuracy deep learning classifier (EfficientNet-B0) to categorize axial brain MRI slices into Glioma, Meningioma, Pituitary Tumor, or Normal (>98% accuracy target).",
                     pt_size=10.5, space_after=10.0)
    add_bullet_point(tf_obj, "2. Deep Segmentation Benchmarking:",
                     "Implement, train, and benchmark 6 semantic segmentation architectures (Standard U-Net, Attention U-Net, Scratch U-Net++, Pretrained ResNet-34 U-Net++, SaberNet, GeoSample) on BRISC 2025.",
                     pt_size=10.5, space_after=10.0)
    add_bullet_point(tf_obj, "3. Morphological Feature Extraction:",
                     "Compute automated quantitative lesion metrics including tumor surface area (mm² and pixel count), perimeter, circularity, and spatial centroid.",
                     pt_size=10.5, space_after=10.0)
    add_bullet_point(tf_obj, "4. Interactive Clinician Dashboard:",
                     "Deploy an intuitive Streamlit interface allowing real-time MRI upload, instant mask overlay, class probability distribution visualization, and diagnostic review.",
                     pt_size=10.5, space_after=0.0)

    # =========================================================================
    # SLIDE 3 (SECTION 2): LITERATURE REVIEW
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "2", "Literature Review", slide_num=3)

    lit_works = [
        ("U-Net: Convolutional Networks for Biomedical Image Segmentation",
         "Ronneberger, Fischer, & Brox (MICCAI 2015)",
         "• Symmetric contracting (encoder) and expanding (decoder) paths connected via long skip connections.\n"
         "• Preserves spatial context across downsampling, serving as the foundational benchmark for medical image segmentation.\n"
         "• Limitation: Large semantic gap between early encoder features and deep decoder features."),

        ("UNet++: A Nested U-Net Architecture for Medical Image Segmentation",
         "Zhou, Siddiquee, Tajbakhsh, & Liang (IEEE TMI 2018)",
         "• Redesigns skip pathways using dense, nested convolutional blocks connecting encoder and decoder feature maps.\n"
         "• Accumulates multi-scale receptive fields at decoder nodes and incorporates deep supervision loss heads.\n"
         "• Significance: Demonstrated substantial gains on boundary-sensitive medical segmentation tasks."),

        ("Attention U-Net: Learning Where to Look for the Pancreas",
         "Oktay et al. (MIDL / arXiv 2018)",
         "• Proposes additive attention gates in skip connections conditioned on decoder gating signals.\n"
         "• Automatically suppresses task-irrelevant background regions while highlighting salient lesion features with minimal parameter overhead.\n"
         "• Significance: Improves sensitivity to small target structures without extra computational complexity."),

        ("EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
         "Tan & Le (ICML 2019)",
         "• Establishes compound scaling, systematically balancing network depth, width, and input image resolution.\n"
         "• EfficientNet-B0 provides competitive ImageNet top-1 accuracy with only 4.01M parameters.\n"
         "• Significance: Ideal candidate for real-time clinical triage, preventing overfitting on moderate medical datasets."),

        ("SaberNet: Self-Attention Based Effective Relation Network",
         "Pattern Recognition (2023)",
         "• Formulates spatial relation matrices (Q·K^T / √d) to capture global pairwise pixel dependencies across the MRI slice.\n"
         "• Significance: Evaluated in our benchmark to assess whether whole-slice self-attention outperforms local convolutions on brain neoplasms."),

        ("GeoSample: Geometry-Guided Sampling Operators for Medical Segmentation",
         "Medical Imaging Benchmark (2025)",
         "• Replaces fixed 3×3 convolution grids with learnable directional boundary probes (orientation and step sizes).\n"
         "• Significance: Explored in our project to evaluate explicit boundary gradient and curvature token extraction on tumor margins."),
    ]

    coords_s3 = [
        (Inches(0.8), Inches(1.55), Inches(3.75), Inches(2.5)),
        (Inches(4.78), Inches(1.55), Inches(3.75), Inches(2.5)),
        (Inches(8.75), Inches(1.55), Inches(3.75), Inches(2.5)),
        (Inches(0.8), Inches(4.3), Inches(3.75), Inches(2.6)),
        (Inches(4.78), Inches(4.3), Inches(3.75), Inches(2.6)),
        (Inches(8.75), Inches(4.3), Inches(3.75), Inches(2.6)),
    ]

    for (title, citation, summary), (lx, ly, lw, lh) in zip(lit_works, coords_s3):
        add_card(s3, lx, ly, lw, lh, bg_color=BG_CARD, border_color=BORDER_CARD)
        tb = s3.shapes.add_textbox(lx + Inches(0.2), ly + Inches(0.12), lw - Inches(0.4), lh - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.name = "Calibri"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY

        pc = tf.add_paragraph()
        pc.text = citation
        pc.font.name = "Calibri"
        pc.font.size = Pt(9.0)
        pc.font.italic = True
        pc.font.color.rgb = SECONDARY_NAVY
        pc.space_after = Pt(4)

        ps = tf.add_paragraph()
        ps.text = summary
        ps.font.name = "Calibri"
        ps.font.size = Pt(9.0)
        ps.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 4 (SECTION 3): METHODOLOGY AND WORK PLAN
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "3", "Methodology and Work Plan", slide_num=4)

    # Left Column: Dataset & Technical Methodology
    add_card(s4, Inches(0.8), Inches(1.55), Inches(5.7), Inches(5.35), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_m1 = s4.shapes.add_textbox(Inches(1.05), Inches(1.7), Inches(5.2), Inches(5.05))
    tf_m1 = tb_m1.text_frame
    tf_m1.word_wrap = True

    p_mh = tf_m1.paragraphs[0]
    p_mh.text = "TECHNICAL METHODOLOGY"
    p_mh.font.name = "Calibri"
    p_mh.font.size = Pt(12.5)
    p_mh.font.bold = True
    p_mh.font.color.rgb = PRIMARY_NAVY
    p_mh.space_after = Pt(8)

    add_bullet_point(tf_m1, "Benchmark Dataset (BRISC 2025):",
                     "6,000 axial brain MRI slices (5,000 train / 1,000 test) across 4 balanced classes: Glioma, Meningioma, Pituitary, Normal. Paired ground-truth binary masks for all tumor slices.",
                     pt_size=10.0, space_after=8.0)
    add_bullet_point(tf_m1, "Preprocessing Pipeline:",
                     "Intensity scaling, format normalization, CLAHE contrast enhancement, and multi-resolution sizing (224×224 for classification, 256×256 & 512×512 for segmentation).",
                     pt_size=10.0, space_after=8.0)
    add_bullet_point(tf_m1, "Classification Engine:",
                     "EfficientNet-B0 with ImageNet pretrained transfer learning, custom classification head, Cross-Entropy Loss, AdamW optimizer, and Cosine Annealing learning rate schedule.",
                     pt_size=10.0, space_after=8.0)
    add_bullet_point(tf_m1, "Segmentation Pipeline:",
                     "Trained 6 architectures with hybrid Dice + Binary Cross-Entropy Loss (DiceBCELoss) to balance pixel-level classification and region overlap. Deep supervision on U-Net++.",
                     pt_size=10.0, space_after=8.0)
    add_bullet_point(tf_m1, "Compute Infrastructure:",
                     "Local NVIDIA RTX 2050 (4 GB) + Cloud NVIDIA Tesla T4 (16 GB) utilizing FP16 Automatic Mixed Precision (AMP) for 2.2× faster training.",
                     pt_size=10.0, space_after=0.0)

    # Right Column: Phased Work Plan
    add_card(s4, Inches(6.83), Inches(1.55), Inches(5.7), Inches(5.35), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_m2 = s4.shapes.add_textbox(Inches(7.08), Inches(1.7), Inches(5.2), Inches(5.05))
    tf_m2 = tb_m2.text_frame
    tf_m2.word_wrap = True

    p_wph = tf_m2.paragraphs[0]
    p_wph.text = "PHASED PROJECT WORK PLAN"
    p_wph.font.name = "Calibri"
    p_wph.font.size = Pt(12.5)
    p_wph.font.bold = True
    p_wph.font.color.rgb = PRIMARY_NAVY
    p_wph.space_after = Pt(8)

    phases = [
        ("Phase 0: Environment & Data Setup", "Dataset acquisition, validation, PyTorch CUDA 12.4 environment configuration.", "Completed"),
        ("Phase 1: Preprocessing Module", "Format validation, CLAHE normalization, dual-tensor transforms.", "Completed"),
        ("Phase 2: Classification Module", "EfficientNet-B0 training, validation, and test evaluation (99.00% accuracy).", "Completed"),
        ("Phase 3: Segmentation Benchmarking", "Trained 6 architectures; selected Pretrained ResNet-34 U-Net++ (87.78% Dice).", "Completed"),
        ("Phase 4: Morphometric Feature Extraction", "Calculating tumor surface area (mm²), perimeter, centroid coordinates.", "Active (70%)"),
        ("Phase 5: Medical Knowledge Base (RAG)", "Vector store compilation (ChromaDB + all-MiniLM) for clinical guideline retrieval.", "Scheduled (Nov)"),
        ("Phase 6: Preliminary Report Generation", "Gemini LLM prompt engineering for structured draft report generation.", "Scheduled (Nov)"),
        ("Phase 7: System Integration & Testing", "Pipeline unification, latency profiling, Streamlit deployment, final documentation.", "Scheduled (Dec)"),
    ]

    for p_name, p_desc, p_status in phases:
        p = tf_m2.add_paragraph()
        p.space_after = Pt(4)
        p.font.name = "Calibri"
        p.font.size = Pt(9.5)

        r_title = p.add_run()
        r_title.text = f"• {p_name}: "
        r_title.font.bold = True
        r_title.font.color.rgb = PRIMARY_NAVY

        r_desc = p.add_run()
        r_desc.text = f"{p_desc} "
        r_desc.font.color.rgb = TEXT_DARK

        r_st = p.add_run()
        r_st.text = f"[{p_status}]"
        r_st.font.bold = True
        r_st.font.color.rgb = PRIMARY_NAVY if p_status == "Completed" else SECONDARY_NAVY

    # =========================================================================
    # SLIDE 5 (SECTION 4): PROGRESS ACHIEVED
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "4", "Progress Achieved", slide_num=5)

    # Top Half: Classification Results Card
    add_card(s5, Inches(0.8), Inches(1.55), Inches(11.73), Inches(2.2), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_prog_clf = s5.shapes.add_textbox(Inches(1.05), Inches(1.65), Inches(11.23), Inches(2.0))
    tf_pc = tb_prog_clf.text_frame
    tf_pc.word_wrap = True

    p_pct = tf_pc.paragraphs[0]
    p_pct.text = "A. TUMOR CLASSIFICATION (EfficientNet-B0 — Evaluated on 1,000 Unseen Test Scans)"
    p_pct.font.name = "Calibri"
    p_pct.font.size = Pt(11.5)
    p_pct.font.bold = True
    p_pct.font.color.rgb = PRIMARY_NAVY
    p_pct.space_after = Pt(6)

    add_bullet_point(tf_pc, "Overall Performance Metrics:",
                     "Achieved 99.00% Test Accuracy, 0.9899 Macro-F1 Score, and 0.0529 Test Loss under the full training protocol. On standard 85/15 validation split: 98.28% Test Accuracy (0.9830 Macro-F1).",
                     pt_size=10.0, space_after=4.0)
    add_bullet_point(tf_pc, "Class-Wise Precision / Recall Breakdown:",
                     "Glioma: 98.8% Precision / 98.4% Recall • Meningioma: 98.4% Precision / 98.0% Recall • Pituitary Tumor: 99.2% Precision / 99.6% Recall • No Tumor (Healthy): 100.0% Precision / 100.0% Recall (Zero False Positives).",
                     pt_size=10.0, space_after=4.0)
    add_bullet_point(tf_pc, "Computational Efficiency:",
                     "Model size is only 4.01 Million parameters; per-slice inference latency is under 15 milliseconds on an edge GPU (NVIDIA RTX 2050), ensuring real-time responsiveness.",
                     pt_size=10.0, space_after=0.0)

    # Bottom Half Left: Segmentation Benchmark Table
    add_card(s5, Inches(0.8), Inches(3.9), Inches(7.5), Inches(3.0), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_tlabel = s5.shapes.add_textbox(Inches(1.0), Inches(3.95), Inches(7.1), Inches(0.3))
    tf_tl = tb_tlabel.text_frame
    tf_tl.word_wrap = True
    pt_tl = tf_tl.paragraphs[0]
    pt_tl.text = "B. 6-MODEL SEGMENTATION BENCHMARK (860 Unseen Test Scans)"
    pt_tl.font.name = "Calibri"
    pt_tl.font.size = Pt(10.5)
    pt_tl.font.bold = True
    pt_tl.font.color.rgb = PRIMARY_NAVY

    # Table
    tbl_shape = s5.shapes.add_table(7, 5, Inches(1.0), Inches(4.3), Inches(7.1), Inches(2.45))
    table = tbl_shape.table
    table.columns[0].width = Inches(2.7)
    table.columns[1].width = Inches(1.0)
    table.columns[2].width = Inches(1.1)
    table.columns[3].width = Inches(1.1)
    table.columns[4].width = Inches(1.2)

    headers = ["Architecture", "Params", "Dice Score", "Mean IoU", "Outcome"]
    for idx, h in enumerate(headers):
        cell = table.cell(0, idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY_NAVY
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Calibri"
        p.font.size = Pt(9.0)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    seg_rows = [
        ("Pretrained ResNet-34 U-Net++", "23.67 M", "87.78%", "81.11%", "Selected (SOTA)"),
        ("Scratch U-Net++", "9.16 M", "87.14%", "80.18%", "Rank 2"),
        ("Attention U-Net", "7.98 M", "86.96%", "79.93%", "Rank 3"),
        ("Standard Baseline U-Net", "7.85 M", "86.84%", "~79.60%", "Baseline"),
        ("SaberNet (Spatial Relations)", "8.38 M", "86.66%", "79.70%", "Rank 5"),
        ("GeoSample U-Net (Probes)", "17.72 M", "86.46%", "~79.20%", "Rank 6"),
    ]

    for r_idx, r_vals in enumerate(seg_rows):
        is_sel = (r_idx == 0)
        bg = HIGHLIGHT_ROW if is_sel else (BG_CARD if r_idx % 2 == 0 else WHITE)
        for c_idx, val in enumerate(r_vals):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"
            p.font.size = Pt(8.5)
            p.font.color.rgb = PRIMARY_NAVY if is_sel else TEXT_DARK
            if c_idx == 0 or is_sel:
                p.font.bold = True
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER

    # Bottom Half Right: Streamlit Interface & Visual Inspection
    add_card(s5, Inches(8.5), Inches(3.9), Inches(4.03), Inches(3.0), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_ui = s5.shapes.add_textbox(Inches(8.7), Inches(3.95), Inches(3.63), Inches(2.85))
    tf_ui = tb_ui.text_frame
    tf_ui.word_wrap = True

    p_uih = tf_ui.paragraphs[0]
    p_uih.text = "C. INTERACTIVE STREAMLIT DASHBOARD"
    p_uih.font.name = "Calibri"
    p_uih.font.size = Pt(10.5)
    p_uih.font.bold = True
    p_uih.font.color.rgb = PRIMARY_NAVY
    p_uih.space_after = Pt(4)

    add_bullet_point(tf_ui, "Functional Deployment (`ui/app.py`):",
                     "Interactive web dashboard allowing direct upload of MRI slices with instant display of prediction KPIs, class probabilities, and boundary overlays.",
                     pt_size=9.0, space_after=4.0)
    add_bullet_point(tf_ui, "Real-Time Contour Rendering:",
                     "Alpha-blended mask visualization over anatomical scan; calculates total lesion pixels and area in mm².",
                     pt_size=9.0, space_after=4.0)
    add_bullet_point(tf_ui, "Singleton Model Caching:",
                     "Models remain preloaded in GPU memory via `MedBoardState`, eliminating cold-start reload latency.",
                     pt_size=9.0, space_after=0.0)

    # =========================================================================
    # SLIDE 6 (SECTION 5): FUTURE WORK AND TIMELINE
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "5", "Future Work and Timeline", slide_num=6)

    # Left Column: Future Work
    add_card(s6, Inches(0.8), Inches(1.55), Inches(5.7), Inches(5.35), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_fw = s6.shapes.add_textbox(Inches(1.05), Inches(1.7), Inches(5.2), Inches(5.05))
    tf_fw = tb_fw.text_frame
    tf_fw.word_wrap = True

    p_fwh = tf_fw.paragraphs[0]
    p_fwh.text = "FUTURE WORK"
    p_fwh.font.name = "Calibri"
    p_fwh.font.size = Pt(12.5)
    p_fwh.font.bold = True
    p_fwh.font.color.rgb = PRIMARY_NAVY
    p_fwh.space_after = Pt(8)

    add_bullet_point(tf_fw, "1. Finalize Morphological Feature Extractor:",
                     "Complete automated geometric calculation of tumor surface area, perimeter, circularity, aspect ratio, and centroid coordinates relative to anatomical midline to assist clinical staging.",
                     pt_size=10.0, space_after=10.0)
    add_bullet_point(tf_fw, "2. RAG Medical Knowledge Base Construction:",
                     "Build a local vector store (ChromaDB + sentence-transformers/all-MiniLM-L6-v2) populated with verified neuro-oncology guidelines (WHO CNS5 classification criteria) to retrieve clinical context for identified tumor types.",
                     pt_size=10.0, space_after=10.0)
    add_bullet_point(tf_fw, "3. LLM-Based Preliminary Report Generation:",
                     "Integrate Gemini API with structured diagnostic prompt templates to synthesize classification, segmentation metrics, and retrieved clinical guidelines into a standardized draft radiology report.",
                     pt_size=10.0, space_after=10.0)
    add_bullet_point(tf_fw, "4. Multi-Agent Pipeline Integration & Validation:",
                     "Unify Preprocessing, Classification, Segmentation, Feature Extraction, RAG, and Reporting agents into an autonomous workflow with clinical safety guardrails and edge-case testing.",
                     pt_size=10.0, space_after=0.0)

    # Right Column: Timeline (Gantt Schedule)
    add_card(s6, Inches(6.83), Inches(1.55), Inches(5.7), Inches(5.35), bg_color=BG_CARD, border_color=BORDER_CARD)
    tb_time = s6.shapes.add_textbox(Inches(7.08), Inches(1.7), Inches(5.2), Inches(5.05))
    tf_time = tb_time.text_frame
    tf_time.word_wrap = True

    p_th = tf_time.paragraphs[0]
    p_th.text = "TIMELINE & MILESTONE ROADMAP"
    p_th.font.name = "Calibri"
    p_th.font.size = Pt(12.5)
    p_th.font.bold = True
    p_th.font.color.rgb = PRIMARY_NAVY
    p_th.space_after = Pt(8)

    timeline_items = [
        ("Weeks 8–9 (Late October — Current):",
         "Mid-Semester Evaluation presentation and progress reporting. Finalize automated lesion feature extraction module (`feature_extractor.py`)."),
        ("Weeks 10–11 (Early–Mid November):",
         "Construct ChromaDB vector store; ingest WHO CNS5 clinical guidelines; implement dense retrieval agent for neuro-oncology context."),
        ("Weeks 12–13 (Late November):",
         "Implement Chief Synthesis Agent using Gemini LLM; design clinical prompt templates; validate factual consistency against hallucinations."),
        ("Weeks 14–15 (December):",
         "End-to-end multi-agent integration and testing; latency profiling; UI polishing; preparation of final project dissertation and defense."),
    ]

    for period, desc in timeline_items:
        p = tf_time.add_paragraph()
        p.space_after = Pt(10)
        p.font.name = "Calibri"
        p.font.size = Pt(10.0)

        r_p = p.add_run()
        r_p.text = f"• {period}\n  "
        r_p.font.bold = True
        r_p.font.color.rgb = PRIMARY_NAVY

        r_d = p.add_run()
        r_d.text = desc
        r_d.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 7: CONCLUSION & Q&A
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)

    # Full Background
    bg_full = s7.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_WIDTH, SLIDE_HEIGHT)
    bg_full.fill.solid()
    bg_full.fill.fore_color.rgb = PRIMARY_NAVY
    bg_full.line.fill.background()

    # Center Heading
    tb_end = s7.shapes.add_textbox(Inches(1.5), Inches(1.2), Inches(10.33), Inches(1.8))
    tf_end = tb_end.text_frame
    tf_end.word_wrap = True

    p_th = tf_end.paragraphs[0]
    p_th.text = "Thank You!"
    p_th.font.name = "Calibri"
    p_th.font.size = Pt(38)
    p_th.font.bold = True
    p_th.font.color.rgb = WHITE
    p_th.alignment = PP_ALIGN.CENTER
    p_th.space_after = Pt(6)

    p_qa = tf_end.add_paragraph()
    p_qa.text = "Questions & Discussion"
    p_qa.font.name = "Calibri"
    p_qa.font.size = Pt(19)
    p_qa.font.color.rgb = RGBColor(191, 219, 254)
    p_qa.alignment = PP_ALIGN.CENTER

    # Bottom Details Card
    add_card(s7, Inches(1.8), Inches(3.4), Inches(9.73), Inches(3.3), bg_color=BG_CARD, border_color=BORDER_CARD)

    tb_inf = s7.shapes.add_textbox(Inches(2.1), Inches(3.6), Inches(9.13), Inches(2.9))
    tf_inf = tb_inf.text_frame
    tf_inf.word_wrap = True

    pi_t = tf_inf.paragraphs[0]
    pi_t.text = "MedBoard: Brain MRI Segmentation and Classification System"
    pi_t.font.name = "Calibri"
    pi_t.font.size = Pt(13)
    pi_t.font.bold = True
    pi_t.font.color.rgb = PRIMARY_NAVY
    pi_t.alignment = PP_ALIGN.CENTER
    pi_t.space_after = Pt(10)

    pi_m = tf_inf.add_paragraph()
    pi_m.text = "Project Team: Pragya Thakur (123102162) • Shivanshu Sajwan (123102193) • Digvijay Singh (123102198)"
    pi_m.font.name = "Calibri"
    pi_m.font.size = Pt(11)
    pi_m.font.bold = True
    pi_m.font.color.rgb = SECONDARY_NAVY
    pi_m.alignment = PP_ALIGN.CENTER
    pi_m.space_after = Pt(8)

    pi_c = tf_inf.add_paragraph()
    pi_c.text = "Project Coordinator / Mentor: Dr. Vishwas Rathi\nEvaluation Committee: Dr. Niyati Baliyan & Dr. Vijay Kumar Sharma"
    pi_c.font.name = "Calibri"
    pi_c.font.size = Pt(10.5)
    pi_c.font.color.rgb = TEXT_DARK
    pi_c.alignment = PP_ALIGN.CENTER
    pi_c.space_after = Pt(8)

    pi_inst = tf_inf.add_paragraph()
    pi_inst.text = "Department of Computer Engineering • National Institute of Technology Kurukshetra"
    pi_inst.font.name = "Calibri"
    pi_inst.font.size = Pt(10)
    pi_inst.font.italic = True
    pi_inst.font.color.rgb = TEXT_MUTED
    pi_inst.alignment = PP_ALIGN.CENTER

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    print(f"✅ Crisp 5-Section Mid-Sem PPT generated successfully at: {output_path}")


if __name__ == "__main__":
    out_p = "docs/MedBoard_MidSem_Official.pptx"
    if len(sys.argv) > 1:
        out_p = sys.argv[1]
    build_presentation(out_p)
