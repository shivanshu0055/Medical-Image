"""
MedBoard — scripts/generate_midsem_ppt.py
Generates the official Mid-Semester Project Evaluation PowerPoint Presentation (.pptx)
for NIT Kurukshetra, Department of Computer Engineering (AY: 2026-27).

Strictly adheres to:
  1. Official Notification Structure (5 Required Sections)
  2. Static, Professional Academic Color Palette (Navy #1A365D, Slate #2D3748, Light Gray #F8FAFC)
  3. Clean, Uniform Slide Architecture and Typography
"""

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Static, Dignified Academic Color Palette ──────────────────────────────────
PRIMARY_NAVY    = RGBColor(26, 54, 93)      # Deep Academic Navy (#1A365D)
SECONDARY_NAVY  = RGBColor(43, 108, 176)    # Academic Steel Blue (#2B6CB0)
TEXT_DARK       = RGBColor(45, 55, 72)      # Charcoal Body Text (#2D3748)
TEXT_MUTED      = RGBColor(113, 128, 150)   # Slate Gray (#718096)
BG_LIGHT        = RGBColor(248, 250, 252)   # Clean Off-White Card Fill (#F8FAFC)
BORDER_SUBTLE   = RGBColor(203, 213, 225)   # Crisp Light Gray Border (#CBD5E0)
BORDER_HEADER   = RGBColor(226, 232, 240)   # Divider Line Gray (#E2E8F0)
WHITE           = RGBColor(255, 255, 255)
CALLOUT_BG      = RGBColor(241, 245, 249)   # Subtle Cool Slate Tint (#F1F5F9)

SLIDE_WIDTH  = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)


def add_slide_header(slide, tracker_text: str, title_text: str, subtitle_text: str = None, slide_num: int = None):
    """
    Standardized, static academic slide header.
    Uniform top tracker, deep navy title, slate subtitle, and horizontal rule.
    """
    # Tracker / Section Eyebrow
    tracker_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.3))
    tf_tr = tracker_box.text_frame
    tf_tr.word_wrap = True
    tf_tr.margin_left = tf_tr.margin_top = tf_tr.margin_right = tf_tr.margin_bottom = 0
    p_tr = tf_tr.paragraphs[0]
    p_tr.text = tracker_text.upper()
    p_tr.font.name = "Calibri"
    p_tr.font.size = Pt(9.5)
    p_tr.font.bold = True
    p_tr.font.color.rgb = SECONDARY_NAVY

    # Title & Subtitle Box
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.75))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_t = tf_title.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = "Calibri"
    p_t.font.size = Pt(21)
    p_t.font.bold = True
    p_t.font.color.rgb = PRIMARY_NAVY

    if subtitle_text:
        p_sub = tf_title.add_paragraph()
        p_sub.text = subtitle_text
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.space_before = Pt(2)

    # Divider line under title
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.48), Inches(11.73), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_HEADER
    line.line.fill.background()

    # Footer note & slide number
    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.1), Inches(10.5), Inches(0.3))
    tf_f = footer_box.text_frame
    tf_f.margin_left = tf_f.margin_top = tf_f.margin_right = tf_f.margin_bottom = 0
    p_f = tf_f.paragraphs[0]
    p_f.text = "NIT Kurukshetra • Department of Computer Engineering • B.Tech Project Work (CSPC401 & ITPC401)"
    p_f.font.name = "Calibri"
    p_f.font.size = Pt(8.5)
    p_f.font.color.rgb = TEXT_MUTED

    if slide_num is not None:
        num_box = slide.shapes.add_textbox(Inches(11.5), Inches(7.1), Inches(1.0), Inches(0.3))
        tf_num = num_box.text_frame
        tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = f"{slide_num} / 16"
        p_num.font.name = "Calibri"
        p_num.font.size = Pt(8.5)
        p_num.font.bold = True
        p_num.font.color.rgb = TEXT_MUTED
        p_num.alignment = PP_ALIGN.RIGHT


def add_card(slide, left, top, width, height, bg_color=BG_LIGHT, border_color=BORDER_SUBTLE):
    """Draws a clean, rounded rectangular card container."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.0)
    else:
        shape.line.fill.background()
    return shape


def add_bullet_item(text_frame, bold_prefix: str, content: str, pt_size: float = 10.5, space_after: float = 5.0):
    """Helper to append a clean bullet point with bold lead."""
    p = text_frame.add_paragraph() if len(text_frame.paragraphs[0].text) > 0 else text_frame.paragraphs[0]
    p.space_after = Pt(space_after)
    p.font.name = "Calibri"
    p.font.size = Pt(pt_size)

    r_bold = p.add_run()
    r_bold.text = bold_prefix + " "
    r_bold.font.bold = True
    r_bold.font.color.rgb = PRIMARY_NAVY

    r_text = p.add_run()
    r_text.text = content
    r_text.font.bold = False
    r_text.font.color.rgb = TEXT_DARK


def build_midsem_presentation(output_path: str = "docs/MedBoard_MidSem_Evaluation.pptx"):
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Clean Academic Title Format)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # Top Primary Banner
    top_rect = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_WIDTH, Inches(3.3))
    top_rect.fill.solid()
    top_rect.fill.fore_color.rgb = PRIMARY_NAVY
    top_rect.line.fill.background()

    # Thin Divider Line
    div_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(3.3), SLIDE_WIDTH, Inches(0.04))
    div_bar.fill.solid()
    div_bar.fill.fore_color.rgb = SECONDARY_NAVY
    div_bar.line.fill.background()

    # Institute & Department Header
    inst_box = s1.shapes.add_textbox(Inches(0.9), Inches(0.4), Inches(11.5), Inches(0.7))
    tf_inst = inst_box.text_frame
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

    # Title & Subtitle in Hero Section
    title_box = s1.shapes.add_textbox(Inches(0.9), Inches(1.25), Inches(11.5), Inches(1.7))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    pt = tf_t.paragraphs[0]
    pt.text = "MedBoard: Multi-Agent Brain MRI Analysis & Automated Reporting System"
    pt.font.name = "Calibri"
    pt.font.size = Pt(25)
    pt.font.bold = True
    pt.font.color.rgb = WHITE
    pt.space_after = Pt(6)

    pt_sub = tf_t.add_paragraph()
    pt_sub.text = "Mid-Semester Project Evaluation • B.Tech Project Work (CSPC401 & ITPC401)"
    pt_sub.font.name = "Calibri"
    pt_sub.font.size = Pt(13.5)
    pt_sub.font.color.rgb = RGBColor(191, 219, 254)

    # Metadata Cards (Bottom Half)
    # Card 1: Team Members
    add_card(s1, Inches(0.9), Inches(3.6), Inches(5.6), Inches(3.2), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)
    box_m = s1.shapes.add_textbox(Inches(1.15), Inches(3.75), Inches(5.1), Inches(2.9))
    tf_m = box_m.text_frame
    tf_m.word_wrap = True
    pm_t = tf_m.paragraphs[0]
    pm_t.text = "PROJECT TEAM MEMBERS"
    pm_t.font.name = "Calibri"
    pm_t.font.size = Pt(11)
    pm_t.font.bold = True
    pm_t.font.color.rgb = PRIMARY_NAVY
    pm_t.space_after = Pt(8)

    members = [
        ("Pragya Thakur", "Roll No: 123102162", "B.Tech Computer Engineering"),
        ("Shivanshu Sajwan", "Roll No: 123102193", "B.Tech Computer Engineering"),
        ("Digvijay Singh", "Roll No: 123102198", "B.Tech Computer Engineering"),
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

    # Card 2: Coordinators & Committee
    add_card(s1, Inches(6.8), Inches(3.6), Inches(5.6), Inches(3.2), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)
    box_c = s1.shapes.add_textbox(Inches(7.05), Inches(3.75), Inches(5.1), Inches(2.9))
    tf_c = box_c.text_frame
    tf_c.word_wrap = True
    pc_t = tf_c.paragraphs[0]
    pc_t.text = "FACULTY COORDINATOR & EVALUATION COMMITTEE"
    pc_t.font.name = "Calibri"
    pc_t.font.size = Pt(11)
    pc_t.font.bold = True
    pc_t.font.color.rgb = PRIMARY_NAVY
    pc_t.space_after = Pt(8)

    add_bullet_item(tf_c, "Project Coordinator / Mentor:", "Dr. Vishwas Rathi", pt_size=11.0, space_after=8.0)
    add_bullet_item(tf_c, "Evaluation Committee Members:", "Dr. Niyati Baliyan\nDr. Vijay Kumar Sharma", pt_size=11.0, space_after=8.0)
    add_bullet_item(tf_c, "Evaluation Schedule:", "Mid-Semester Evaluation • Department of Computer Engineering", pt_size=10.0, space_after=0.0)

    # =========================================================================
    # SLIDE 2: PRESENTATION AGENDA
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "EXECUTIVE AGENDA", "Presentation Structure & Notification Compliance",
                     "Structured strictly in accordance with Department of Computer Engineering Mid-Sem Guidelines", slide_num=2)

    agenda_items = [
        ("01", "Problem Identification & Objectives",
         "Clinical neuro-oncology challenges, diagnostic latency, limitations of standalone CAD systems, and primary MedBoard objectives."),
        ("02", "Literature Review & Research Gaps",
         "Survey of foundational architectures (U-Net, U-Net++, Attention U-Net, EfficientNet, SaberNet, GeoSample) and key research gaps."),
        ("03", "Methodology & Work Plan",
         "End-to-end multi-agent pipeline architecture, BRISC 2025 benchmark dataset, dual-compute infrastructure, and experimental setup."),
        ("04", "Progress Achieved (Experimental Results)",
         "99.00% classification accuracy on BRISC 2025; 6-model segmentation benchmark (ResNet-34 U-Net++ top at 87.78% Dice); Streamlit UI."),
        ("05", "Future Work & Project Timeline",
         "Gantt schedule for feature extraction, RAG medical guideline retrieval, Gemini LLM reporting agent, and final thesis integration."),
    ]

    top_y = 1.7
    for num, heading, desc in agenda_items:
        add_card(s2, Inches(0.8), Inches(top_y), Inches(11.73), Inches(0.88), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        # Number Badge (Static Navy)
        nb = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(top_y + 0.15), Inches(0.8), Inches(0.58))
        nb.fill.solid()
        nb.fill.fore_color.rgb = PRIMARY_NAVY
        nb.line.fill.background()
        tf_nb = nb.text_frame
        p_nb = tf_nb.paragraphs[0]
        p_nb.text = num
        p_nb.font.name = "Calibri"
        p_nb.font.size = Pt(15)
        p_nb.font.bold = True
        p_nb.font.color.rgb = WHITE
        p_nb.alignment = PP_ALIGN.CENTER

        # Content Box
        tbox = s2.shapes.add_textbox(Inches(2.0), Inches(top_y + 0.08), Inches(10.3), Inches(0.72))
        tf = tbox.text_frame
        tf.word_wrap = True
        p_h = tf.paragraphs[0]
        p_h.text = heading
        p_h.font.name = "Calibri"
        p_h.font.size = Pt(12.5)
        p_h.font.bold = True
        p_h.font.color.rgb = PRIMARY_NAVY

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.name = "Calibri"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = TEXT_DARK

        top_y += 1.02

    # =========================================================================
    # SLIDE 3: PROBLEM IDENTIFICATION
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "SECTION 1: PROBLEM IDENTIFICATION", "Clinical Dilemmas in Brain MRI Diagnostics",
                     "Overcoming diagnostic delays, infiltrative complexity, and black-box barriers in neuro-oncology", slide_num=3)

    prob_cards = [
        ("High Diagnostic Workload & Triage Latency",
         "Over 300,000 primary CNS neoplasms are diagnosed globally each year. The severe shortage of specialized neuroradiologists causes diagnostic bottlenecks where every millimeter of tumor progression affects surgical margins and survival."),
        ("Infiltrative & Ambiguous Lesion Boundaries",
         "Intracranial neoplasms—particularly high-grade Gliomas and atypical Meningiomas—exhibit diffuse infiltrative edges with minimal intensity contrast against normal brain parenchyma, making manual boundary tracing inconsistent."),
        ("The 'Black-Box' Perception Barrier",
         "Conventional deep learning CAD algorithms output raw classification labels or binary masks without clinical rationale, quantitative morphological biomarkers (area, perimeter, density), or medical guideline citations necessary for clinical adoption."),
        ("Fragmented Diagnostic Workflows",
         "Clinical radiology today relies on disparate tools: one utility for detection, manual electronic calipers for tumor sizing, and unlinked dictation systems. An integrated, multi-agent perception-to-report pipeline is critically missing."),
    ]

    card_coords = [
        (Inches(0.8), Inches(1.7), Inches(5.7), Inches(2.45)),
        (Inches(6.83), Inches(1.7), Inches(5.7), Inches(2.45)),
        (Inches(0.8), Inches(4.35), Inches(5.7), Inches(2.45)),
        (Inches(6.83), Inches(4.35), Inches(5.7), Inches(2.45)),
    ]

    for (title, desc), (cx, cy, cw, ch) in zip(prob_cards, card_coords):
        add_card(s3, cx, cy, cw, ch, bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        tb = s3.shapes.add_textbox(cx + Inches(0.25), cy + Inches(0.18), cw - Inches(0.5), ch - Inches(0.36))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.name = "Calibri"
        pt.font.size = Pt(12.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY
        pt.space_after = Pt(8)

        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.name = "Calibri"
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 4: PROJECT OBJECTIVES
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "SECTION 1: OBJECTIVES", "Primary Research Goal & Technical Deliverables",
                     "MedBoard: Autonomous Multi-Agent Brain MRI Analysis and Structured Reporting Pipeline", slide_num=4)

    # Primary Goal Callout Card
    add_card(s4, Inches(0.8), Inches(1.7), Inches(11.73), Inches(1.15), bg_color=CALLOUT_BG, border_color=SECONDARY_NAVY)
    tb_goal = s4.shapes.add_textbox(Inches(1.05), Inches(1.78), Inches(11.2), Inches(0.95))
    tf_goal = tb_goal.text_frame
    tf_goal.word_wrap = True

    pg_t = tf_goal.paragraphs[0]
    pg_t.text = "PRIMARY RESEARCH OBJECTIVE"
    pg_t.font.name = "Calibri"
    pg_t.font.size = Pt(11)
    pg_t.font.bold = True
    pg_t.font.color.rgb = SECONDARY_NAVY
    pg_t.space_after = Pt(2)

    pg_b = tf_goal.add_paragraph()
    pg_b.text = "To design and evaluate MedBoard, an end-to-end multi-agent AI framework that ingests raw 2D brain MRI slices, executes high-accuracy tumor classification and sub-millimeter semantic segmentation, extracts morphological biomarkers, and synthesizes guideline-grounded, verifiable preliminary diagnostic reports."
    pg_b.font.name = "Calibri"
    pg_b.font.size = Pt(10.5)
    pg_b.font.color.rgb = TEXT_DARK

    # 4 Pillar Technical Objectives
    pillars = [
        ("1. Automated Classification",
         "Classify axial brain MRI slices into 4 distinct clinical categories (Glioma, Meningioma, Pituitary, Normal) with >98% accuracy using deep transfer learning (EfficientNet-B0)."),
        ("2. Precision Segmentation",
         "Benchmark and optimize 6 semantic segmentation architectures (U-Net++, SaberNet, GeoSample, Attention U-Net) to achieve >86% Mean Dice on complex lesion margins."),
        ("3. Morphometric Biomarkers",
         "Calculate quantitative radiological biomarkers including 2D lesion surface area (mm²), perimeter, equivalent diameter, and centroid coordinates to quantify tumor burden."),
        ("4. Grounded Synthesis & UI",
         "Develop a RAG-powered Chief Synthesis Agent (Gemini 2.0 Flash + ChromaDB) generating standardized preliminary reports with verifiable medical citations and interactive clinician UI."),
    ]

    card_w = Inches(2.78)
    card_h = Inches(3.7)
    lefts = [Inches(0.8), Inches(3.78), Inches(6.76), Inches(9.74)]

    for (p_title, p_desc), cl in zip(pillars, lefts):
        add_card(s4, cl, Inches(3.05), card_w, card_h, bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        # Header Strip
        h_strip = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, cl, Inches(3.05), card_w, Inches(0.06))
        h_strip.fill.solid()
        h_strip.fill.fore_color.rgb = PRIMARY_NAVY
        h_strip.line.fill.background()

        tb_p = s4.shapes.add_textbox(cl + Inches(0.2), Inches(3.22), card_w - Inches(0.4), card_h - Inches(0.4))
        tf_p = tb_p.text_frame
        tf_p.word_wrap = True
        pt = tf_p.paragraphs[0]
        pt.text = p_title
        pt.font.name = "Calibri"
        pt.font.size = Pt(11.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY
        pt.space_after = Pt(8)

        pd = tf_p.add_paragraph()
        pd.text = p_desc
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 5: LITERATURE REVIEW
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "SECTION 2: LITERATURE REVIEW", "Foundational Deep Learning & Medical Imaging Paradigms",
                     "Comparative synthesis of baseline architectures, recent innovations, and theoretical foundations", slide_num=5)

    lit_cards = [
        ("U-Net: Biomedical Image Segmentation", "Ronneberger et al. (MICCAI 2015)",
         "Proposed the symmetric contracting-expanding encoder-decoder network with long skip connections, establishing the fundamental baseline for medical image semantic segmentation."),
        ("U-Net++: Nested Dense Skip Pathways", "Zhou et al. (IEEE TMI 2018)",
         "Redesigned skip connections as dense nested convolutional pathways, mitigating the semantic gap between encoder and decoder feature maps and introducing multi-scale deep supervision."),
        ("Attention U-Net: Additive Gating", "Oktay et al. (MIDL 2018)",
         "Integrated additive soft attention gates within skip pathways to automatically suppress task-irrelevant background regions while accentuating salient tumor features."),
        ("EfficientNet: Compound Scaling", "Tan & Le (ICML 2019)",
         "Systematically balances network depth, width, and input resolution via a fixed compound scaling coefficient, achieving top-tier accuracy with 4.01M parameters (EfficientNet-B0)."),
        ("SaberNet: Spatial Relation Modeling", "Pattern Recognition (2023)",
         "Self-Attention Based Effective Relation Network that constructs pairwise pixel spatial relation matrices (Q·K^T/√d) across the slice to model long-range context."),
        ("GeoSample: Geometry-Guided Sampling", "Medical Imaging Benchmark (2025)",
         "Replaces rigid 3×3 grid convolutions with learned directional boundary probes (orientation θ, step sizes r) to extract explicit gradient and curvature differential tokens."),
    ]

    coords_s5 = [
        (Inches(0.8), Inches(1.7), Inches(3.75), Inches(2.45)),
        (Inches(4.78), Inches(1.7), Inches(3.75), Inches(2.45)),
        (Inches(8.75), Inches(1.7), Inches(3.75), Inches(2.45)),
        (Inches(0.8), Inches(4.35), Inches(3.75), Inches(2.45)),
        (Inches(4.78), Inches(4.35), Inches(3.75), Inches(2.45)),
        (Inches(8.75), Inches(4.35), Inches(3.75), Inches(2.45)),
    ]

    for (title, citation, summary), (lx, ly, lw, lh) in zip(lit_cards, coords_s5):
        add_card(s5, lx, ly, lw, lh, bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        tb = s5.shapes.add_textbox(lx + Inches(0.2), ly + Inches(0.15), lw - Inches(0.4), lh - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = title
        pt.font.name = "Calibri"
        pt.font.size = Pt(11.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY

        pc = tf.add_paragraph()
        pc.text = citation
        pc.font.name = "Calibri"
        pc.font.size = Pt(9.5)
        pc.font.italic = True
        pc.font.color.rgb = SECONDARY_NAVY
        pc.space_after = Pt(6)

        ps = tf.add_paragraph()
        ps.text = summary
        ps.font.name = "Calibri"
        ps.font.size = Pt(9.5)
        ps.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 6: RESEARCH GAPS & MEDBOARD SOLUTIONS
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "SECTION 2: RESEARCH GAPS", "Identified Limitations in Current Literature & MedBoard Solutions",
                     "Bridging the disconnect between computer vision benchmarks and clinical radiology utility", slide_num=6)

    gaps = [
        ("GAP 1: Isolated Diagnostic Tasks",
         "Existing literature isolates classification from segmentation. A radiologist receives either a category label or a raw mask, but rarely an aligned dual-tier assessment.",
         "MedBoard Solution: Unified pipeline executing parallel perception (ResNet-34 U-Net++ segmentation + EfficientNet-B0 classification) synchronized into a shared patient state."),

        ("GAP 2: Lack of Grounded Clinical Explanations",
         "Deep learning models function as black-box predictors, offering no reference to standard medical staging guidelines (e.g., WHO CNS5, RANO criteria).",
         "MedBoard Solution: Integration of a Retrieval-Augmented Generation (RAG) agent matching extracted tumor attributes with a verified medical knowledge base (ChromaDB)."),

        ("GAP 3: Complex Boundary Delineation Failures",
         "Standard convolutional kernels struggle to capture infiltrative micro-margins and fine dural attachments in gliomas and meningiomas.",
         "MedBoard Solution: Benchmarked 6 diverse paradigms (including dense nested skips, attention gates, and geometric sampling); Pretrained ResNet-34 U-Net++ achieved 87.78% Dice."),

        ("GAP 4: Clinician-AI Interface Disconnect",
         "Research codebases often lack real-time deployment interfaces suitable for rapid clinical review and visual verification.",
         "MedBoard Solution: Interactive Streamlit dashboard with real-time contour overlays, class probability graphs, quantitative lesion KPIs, and report exports."),
    ]

    y_gap = 1.7
    for num_title, gap_desc, sol_desc in gaps:
        add_card(s6, Inches(0.8), Inches(y_gap), Inches(11.73), Inches(1.15), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        # Left subtle accent bar
        l_bar = s6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(y_gap), Inches(0.08), Inches(1.15))
        l_bar.fill.solid()
        l_bar.fill.fore_color.rgb = PRIMARY_NAVY
        l_bar.line.fill.background()

        tb = s6.shapes.add_textbox(Inches(1.05), Inches(y_gap + 0.08), Inches(11.3), Inches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = num_title
        pt.font.name = "Calibri"
        pt.font.size = Pt(11.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY
        pt.space_after = Pt(2)

        pg = tf.add_paragraph()
        r_gl = pg.add_run()
        r_gl.text = "Identified Limitation: "
        r_gl.font.bold = True
        r_gl.font.color.rgb = SECONDARY_NAVY
        r_gt = pg.add_run()
        r_gt.text = gap_desc
        r_gt.font.color.rgb = TEXT_DARK
        pg.font.name = "Calibri"
        pg.font.size = Pt(9.5)
        pg.space_after = Pt(2)

        ps = tf.add_paragraph()
        r_sl = ps.add_run()
        r_sl.text = "MedBoard Solution: "
        r_sl.font.bold = True
        r_sl.font.color.rgb = PRIMARY_NAVY
        r_st = ps.add_run()
        r_st.text = sol_desc
        r_st.font.color.rgb = TEXT_DARK
        ps.font.name = "Calibri"
        ps.font.size = Pt(9.5)

        y_gap += 1.3

    # =========================================================================
    # SLIDE 7: METHODOLOGY - SYSTEM ARCHITECTURE
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "SECTION 3: METHODOLOGY", "End-to-End Multi-Agent System Architecture",
                     "Hierarchical agentic workflow chaining image preprocessing, perception, feature extraction, RAG, and LLM synthesis", slide_num=7)

    stages = [
        ("STAGE 1: Preprocessing Agent",
         "• Format validation (.jpg, .png, DICOM)\n• Intensity standardization & CLAHE\n• Dual tensor resizing (224x224 / 256x256)\n• ImageNet normalization stats"),
        ("STAGE 2A: Segmentation Agent",
         "• Pretrained ResNet-34 U-Net++\n• 5-Level dense nested skip pathways\n• Multi-scale deep supervision heads\n• Outputs high-res binary tumor mask"),
        ("STAGE 2B: Classification Agent",
         "• EfficientNet-B0 backbone (4.01M params)\n• 4 Classes: Glioma, Meningioma, Pituitary, Normal\n• Softmax class probabilities\n• 99.00% Verified Test Accuracy"),
        ("STAGE 3: Feature Extraction Agent",
         "• 2D Lesion Surface Area (mm² & pixels)\n• Tumor perimeter, roundness & circularity\n• Sellar / Midline centroid localization\n• Radiomic intensity statistics"),
        ("STAGE 4: Evidence Agent (RAG)",
         "• ChromaDB vector database index\n• all-MiniLM-L6-v2 dense embeddings\n• Neuro-oncology guidelines store\n• Top-k context snippet retrieval"),
        ("STAGE 5: Chief Synthesis Agent",
         "• Gemini 2.0 Flash API integration\n• Structured clinical radiology reporting\n• Verifiable clinical explanations\n• Streamlit clinician dashboard frontend"),
    ]

    card_coords_s7 = [
        (Inches(0.8), Inches(1.7), Inches(3.75), Inches(2.45)),
        (Inches(4.78), Inches(1.7), Inches(3.75), Inches(2.45)),
        (Inches(8.75), Inches(1.7), Inches(3.75), Inches(2.45)),
        (Inches(0.8), Inches(4.35), Inches(3.75), Inches(2.45)),
        (Inches(4.78), Inches(4.35), Inches(3.75), Inches(2.45)),
        (Inches(8.75), Inches(4.35), Inches(3.75), Inches(2.45)),
    ]

    for (s_title, s_bullets), (sx, sy, sw, sh) in zip(stages, card_coords_s7):
        add_card(s7, sx, sy, sw, sh, bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        tb = s7.shapes.add_textbox(sx + Inches(0.2), sy + Inches(0.15), sw - Inches(0.4), sh - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = s_title
        pt.font.name = "Calibri"
        pt.font.size = Pt(11.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY
        pt.space_after = Pt(8)

        pb = tf.add_paragraph()
        pb.text = s_bullets
        pb.font.name = "Calibri"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 8: BENCHMARK DATASET & TRAINING SETUP
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "SECTION 3: METHODOLOGY", "Benchmark Dataset & Experimental Configuration",
                     "BRISC 2025 Brain MRI Benchmark and Dual-Tier Compute Infrastructure", slide_num=8)

    # Left Column: Dataset Details
    add_card(s8, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)
    tb_d = s8.shapes.add_textbox(Inches(1.05), Inches(1.85), Inches(5.2), Inches(4.8))
    tf_d = tb_d.text_frame
    tf_d.word_wrap = True

    pd_t = tf_d.paragraphs[0]
    pd_t.text = "BRISC 2025 BENCHMARK DATASET"
    pd_t.font.name = "Calibri"
    pd_t.font.size = Pt(12.5)
    pd_t.font.bold = True
    pd_t.font.color.rgb = PRIMARY_NAVY
    pd_t.space_after = Pt(8)

    add_bullet_item(tf_d, "Benchmark Source:", "Nature Scientific Data / arXiv:2506.14318 benchmark dataset.", pt_size=10.0)
    add_bullet_item(tf_d, "Total Cohort:", "6,000 total axial 2D brain MRI slices (5,000 train / 1,000 test).", pt_size=10.0)
    add_bullet_item(tf_d, "Class Distribution:", "4 balanced clinical categories:\n  • Glioma (1,500 slices)\n  • Meningioma (1,500 slices)\n  • Pituitary Tumor (1,500 slices)\n  • No Tumor / Normal Controls (1,500 slices)", pt_size=10.0)
    add_bullet_item(tf_d, "Ground Truth Masks:", "Pixel-level manual segmentations annotated by expert neuro-radiology panels for all tumor-bearing cases.", pt_size=10.0)
    add_bullet_item(tf_d, "Data Split Hygiene:", "Standardized 90/10 train-validation protocol with zero patient leakage; strictly held-out 1,000-slice test corpus.", pt_size=10.0)

    # Right Column: Training & Compute Pipeline
    add_card(s8, Inches(6.83), Inches(1.7), Inches(5.7), Inches(5.1), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)
    tb_e = s8.shapes.add_textbox(Inches(7.08), Inches(1.85), Inches(5.2), Inches(4.8))
    tf_e = tb_e.text_frame
    tf_e.word_wrap = True

    pe_t = tf_e.paragraphs[0]
    pe_t.text = "TRAINING & COMPUTE SETUP"
    pe_t.font.name = "Calibri"
    pe_t.font.size = Pt(12.5)
    pe_t.font.bold = True
    pe_t.font.color.rgb = PRIMARY_NAVY
    pe_t.space_after = Pt(8)

    add_bullet_item(tf_e, "Hardware Environments:", "Local NVIDIA GeForce RTX 2050 (4 GB VRAM) & Cloud NVIDIA Tesla T4 (16 GB VRAM, Google Colab).", pt_size=10.0)
    add_bullet_item(tf_e, "Software Framework:", "PyTorch 2.5 + CUDA 12.4, torchvision, timm, albumentations.", pt_size=10.0)
    add_bullet_item(tf_e, "Loss Formulations:", "Classification: Cross-Entropy with label smoothing.\nSegmentation: Hybrid Dice + Binary Cross-Entropy Loss.", pt_size=10.0)
    add_bullet_item(tf_e, "Optimization Policy:", "AdamW / Adam with Cosine Annealing learning rate schedule (warm restarts); early stopping patience = 5–10 epochs.", pt_size=10.0)
    add_bullet_item(tf_e, "Automatic Mixed Precision:", "FP16 AMP enabled across all training passes for 2.2x training acceleration and reduced memory overhead.", pt_size=10.0)

    # =========================================================================
    # SLIDE 9: PROGRESS ACHIEVED - CLASSIFICATION
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "SECTION 4: PROGRESS ACHIEVED", "Tumor Classification Performance (EfficientNet-B0)",
                     "99.00% Test Accuracy across 1,000 Unseen Brain MRI Scans on BRISC 2025 Benchmark", slide_num=9)

    # Top KPI Metric Cards (Uniform Static Styling)
    kpis = [
        ("99.00%", "TEST ACCURACY", "Full 5,000 training protocol (4.01M params)"),
        ("0.9899", "MACRO-F1 SCORE", "Balanced precision/recall across all 4 classes"),
        ("0.0529", "TEST LOSS", "Cross-entropy loss on unseen 1,000 scans"),
        ("< 15 ms", "INFERENCE LATENCY", "Real-time diagnostic triage on edge GPU"),
    ]

    kpi_w = Inches(2.78)
    lefts_kpi = [Inches(0.8), Inches(3.78), Inches(6.76), Inches(9.74)]

    for (k_val, k_title, k_desc), kl in zip(kpis, lefts_kpi):
        add_card(s9, kl, Inches(1.7), kpi_w, Inches(1.5), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        tb_k = s9.shapes.add_textbox(kl + Inches(0.1), Inches(1.78), kpi_w - Inches(0.2), Inches(1.3))
        tf_k = tb_k.text_frame
        tf_k.word_wrap = True

        p_val = tf_k.paragraphs[0]
        p_val.text = k_val
        p_val.font.name = "Calibri"
        p_val.font.size = Pt(25)
        p_val.font.bold = True
        p_val.font.color.rgb = PRIMARY_NAVY
        p_val.alignment = PP_ALIGN.CENTER

        p_tit = tf_k.add_paragraph()
        p_tit.text = k_title
        p_tit.font.name = "Calibri"
        p_tit.font.size = Pt(9.5)
        p_tit.font.bold = True
        p_tit.font.color.rgb = SECONDARY_NAVY
        p_tit.alignment = PP_ALIGN.CENTER

        p_sub = tf_k.add_paragraph()
        p_sub.text = k_desc
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(8.5)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.alignment = PP_ALIGN.CENTER

    # Detailed Table of Classification Results
    add_card(s9, Inches(0.8), Inches(3.45), Inches(11.73), Inches(3.35), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

    tbl_shape = s9.shapes.add_table(5, 5, Inches(1.05), Inches(3.6), Inches(11.23), Inches(3.0))
    table = tbl_shape.table
    table.columns[0].width = Inches(2.83)
    table.columns[1].width = Inches(2.1)
    table.columns[2].width = Inches(2.1)
    table.columns[3].width = Inches(2.1)
    table.columns[4].width = Inches(2.1)

    headers = ["Diagnostic Class", "Precision", "Recall", "F1-Score", "Support (Test Set)"]
    for idx, h in enumerate(headers):
        cell = table.cell(0, idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY_NAVY
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    clf_rows = [
        ("Glioma (Infiltrative Intra-axial)", "98.8%", "98.4%", "0.986", "250 scans"),
        ("Meningioma (Extra-axial Dural)", "98.4%", "98.0%", "0.982", "250 scans"),
        ("Pituitary Tumor (Sellar Region)", "99.2%", "99.6%", "0.994", "250 scans"),
        ("No Tumor (Healthy Control)", "100.0%", "100.0%", "1.000", "250 scans"),
    ]

    for row_idx, r_data in enumerate(clf_rows):
        bg = BG_LIGHT if row_idx % 2 == 0 else WHITE
        for col_idx, val in enumerate(r_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_DARK
            if col_idx == 0:
                p.font.bold = True
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 10: PROGRESS ACHIEVED - SEGMENTATION BENCHMARK (6 ARCHITECTURES)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "SECTION 4: PROGRESS ACHIEVED", "Comprehensive 6-Model Segmentation Benchmark",
                     "Evaluated across 860 Unseen Brain MRI Test Scans (BRISC 2025 Benchmark)", slide_num=10)

    # Left: Benchmark Table
    add_card(s10, Inches(0.8), Inches(1.7), Inches(7.5), Inches(5.1), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

    tbl_s10 = s10.shapes.add_table(7, 5, Inches(0.95), Inches(1.85), Inches(7.2), Inches(4.7))
    t10 = tbl_s10.table
    t10.columns[0].width = Inches(2.6)
    t10.columns[1].width = Inches(1.1)
    t10.columns[2].width = Inches(1.1)
    t10.columns[3].width = Inches(1.1)
    t10.columns[4].width = Inches(1.3)

    seg_headers = ["Architecture", "Parameters", "Dice Score", "Mean IoU", "Performance Rank"]
    for idx, h in enumerate(seg_headers):
        cell = t10.cell(0, idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY_NAVY
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Calibri"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    seg_models = [
        ("Pretrained ResNet-34 U-Net++", "23.67 M", "87.78%", "81.11%", "★ SOTA Top Performer"),
        ("Scratch U-Net++", "9.16 M", "87.14%", "80.18%", "Rank 2"),
        ("Attention U-Net", "7.98 M", "86.96%", "79.93%", "Rank 3"),
        ("Standard Baseline U-Net", "7.85 M", "86.84%", "~79.60%", "Baseline Benchmark"),
        ("SaberNet (Spatial Relations)", "8.38 M", "86.66%", "79.70%", "Rank 5"),
        ("GeoSample U-Net (Probes)", "17.72 M", "86.46%", "~79.20%", "Rank 6"),
    ]

    for row_idx, r_data in enumerate(seg_models):
        is_top = (row_idx == 0)
        bg = CALLOUT_BG if is_top else (BG_LIGHT if row_idx % 2 == 0 else WHITE)
        for col_idx, val in enumerate(r_data):
            cell = t10.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"
            p.font.size = Pt(9)
            p.font.color.rgb = PRIMARY_NAVY if is_top else TEXT_DARK
            if col_idx == 0 or is_top:
                p.font.bold = True
            p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    # Right: Visual Qualitative Verification Image
    add_card(s10, Inches(8.5), Inches(1.7), Inches(4.03), Inches(5.1), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)
    tb_img_cap = s10.shapes.add_textbox(Inches(8.65), Inches(1.85), Inches(3.73), Inches(0.5))
    tf_ic = tb_img_cap.text_frame
    tf_ic.word_wrap = True
    p_ic = tf_ic.paragraphs[0]
    p_ic.text = "QUALITATIVE VERIFICATION"
    p_ic.font.name = "Calibri"
    p_ic.font.size = Pt(11)
    p_ic.font.bold = True
    p_ic.font.color.rgb = PRIMARY_NAVY

    # Insert Image
    img_path = Path("seg_result.png")
    if img_path.exists():
        s10.shapes.add_picture(str(img_path), Inches(8.65), Inches(2.4), width=Inches(3.73))

    tb_bot = s10.shapes.add_textbox(Inches(8.65), Inches(4.15), Inches(3.73), Inches(2.5))
    tf_bot = tb_bot.text_frame
    tf_bot.word_wrap = True
    p_b1 = tf_bot.paragraphs[0]
    p_b1.text = "Qualitative Clinical Observations:"
    p_b1.font.name = "Calibri"
    p_b1.font.size = Pt(9.5)
    p_b1.font.bold = True
    p_b1.font.color.rgb = PRIMARY_NAVY
    p_b1.space_after = Pt(4)

    add_bullet_item(tf_bot, "Boundary Conformance:", "Pretrained ResNet-34 U-Net++ captures fine infiltrative dural edges without false-positive leakage into ventricles.", pt_size=8.5, space_after=3.0)
    add_bullet_item(tf_bot, "Nested Skips:", "Dramatically reduces false negatives along irregular margin gradients compared to baseline U-Net.", pt_size=8.5, space_after=0.0)

    # =========================================================================
    # SLIDE 11: PROGRESS ACHIEVED - ARCHITECTURAL INSIGHTS
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "SECTION 4: PROGRESS ACHIEVED", "In-Depth Architectural Analysis & Technical Insights",
                     "Evaluating why specific mechanisms excelled in brain tumor semantic segmentation", slide_num=11)

    arch_insights = [
        ("1. Pretrained ResNet-34 U-Net++ (Top Performer: 87.78% Dice)",
         "• Leveraging ImageNet pretraining endowed early layers with robust edge and gradient filters.\n"
         "• Dilated convolutions in layer4 preserved 16x16 resolution without subsampling artifacts.\n"
         "• Deep supervision at X_0,1, X_0,2, X_0,3, X_0,4 enabled smooth multi-scale gradient flow.\n"
         "• Two-speed optimization (gentle encoder_lr, fast decoder_lr) prevented catastrophic forgetting."),

        ("2. Scratch U-Net++ vs. Baseline U-Net (87.14% vs 86.84%)",
         "• Redesigned skip pathways bridge semantic gap between encoder/decoder stages.\n"
         "• Receptive fields of varying sizes are aggregated at decoder nodes via dense concatenation.\n"
         "• Conclusively proves dense nested skip architecture outperforms standard long skips by +0.30% Dice even from scratch."),

        ("3. Attention U-Net & SaberNet (86.96% & 86.66% Dice)",
         "• Attention U-Net: Additive attention gates suppressed non-tumor brain tissue effectively with minimal parameter overhead (+0.13M params).\n"
         "• SaberNet: Global relation matrices captured whole-slice context but suffered slightly on fine sub-millimeter edge contours due to quadratic spatial pooling."),

        ("4. GeoSample U-Net & High-Resolution (512x512) Exploration",
         "• GeoSample: Directional boundary probes achieved competitive 86.46% Dice using novel differential tokens (gradient, curvature).\n"
         "• 512x512 High-Res: Integrated gradient accumulation to enable memory-efficient training on RTX 2050 without out-of-memory errors."),
    ]

    c_coords_s11 = [
        (Inches(0.8), Inches(1.7), Inches(5.7), Inches(2.45)),
        (Inches(6.83), Inches(1.7), Inches(5.7), Inches(2.45)),
        (Inches(0.8), Inches(4.35), Inches(5.7), Inches(2.45)),
        (Inches(6.83), Inches(4.35), Inches(5.7), Inches(2.45)),
    ]

    for (a_title, a_bullets), (ax, ay, aw, ah) in zip(arch_insights, c_coords_s11):
        add_card(s11, ax, ay, aw, ah, bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        tb = s11.shapes.add_textbox(ax + Inches(0.2), ay + Inches(0.15), aw - Inches(0.4), ah - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = a_title
        pt.font.name = "Calibri"
        pt.font.size = Pt(11.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY
        pt.space_after = Pt(6)

        pb = tf.add_paragraph()
        pb.text = a_bullets
        pb.font.name = "Calibri"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 12: PROGRESS ACHIEVED - STREAMLIT DASHBOARD
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "SECTION 4: PROGRESS ACHIEVED", "Interactive Clinician Decision-Support Interface",
                     "Streamlit Web Application (`ui/app.py`) for Real-Time MRI Inference & Inspection", slide_num=12)

    # Left Column: Features & Workflow
    add_card(s12, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.1), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)
    tb_ui = s12.shapes.add_textbox(Inches(1.05), Inches(1.85), Inches(5.2), Inches(4.8))
    tf_ui = tb_ui.text_frame
    tf_ui.word_wrap = True

    pu_t = tf_ui.paragraphs[0]
    pu_t.text = "FRONTEND WORKFLOW & CLINICAL CAPABILITIES"
    pu_t.font.name = "Calibri"
    pu_t.font.size = Pt(12.5)
    pu_t.font.bold = True
    pu_t.font.color.rgb = PRIMARY_NAVY
    pu_t.space_after = Pt(8)

    add_bullet_item(tf_ui, "Instant Scan Ingestion:", "Supports direct upload of 2D axial MRI slices (PNG, JPG, DICOM) with instant preprocessing and intensity validation.", pt_size=10.0)
    add_bullet_item(tf_ui, "Contour & Mask Overlay:", "Interactive alpha-blended visualization displaying exact tumor boundaries over anatomical brain tissue.", pt_size=10.0)
    add_bullet_item(tf_ui, "Clinical KPI Scorecards:", "Real-time cards presenting:\n  • Predicted Diagnosis (Glioma / Meningioma / Pituitary / Normal)\n  • Classification Confidence (%)\n  • Lesion Sizing (Total pixel area & mm²)\n  • Tumor Presence Binary Flag", pt_size=10.0)
    add_bullet_item(tf_ui, "Probabilistic Distribution:", "Interactive probability bar charts displaying confidence across all 4 candidate diagnostic categories.", pt_size=10.0)
    add_bullet_item(tf_ui, "Radiologist Review Readiness:", "Designed with clean clinical typography and safety disclaimers for research verification.", pt_size=10.0)

    # Right Column: Architecture & Pipeline Integration
    add_card(s12, Inches(6.83), Inches(1.7), Inches(5.7), Inches(5.1), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)
    tb_uip = s12.shapes.add_textbox(Inches(7.08), Inches(1.85), Inches(5.2), Inches(4.8))
    tf_uip = tb_uip.text_frame
    tf_uip.word_wrap = True

    pp_t = tf_uip.paragraphs[0]
    pp_t.text = "BACKEND PIPELINE INTEGRATION (`pipeline.py`)"
    pp_t.font.name = "Calibri"
    pp_t.font.size = Pt(12.5)
    pp_t.font.bold = True
    pp_t.font.color.rgb = PRIMARY_NAVY
    pp_t.space_after = Pt(8)

    add_bullet_item(tf_uip, "Shared State Data Class (`MedBoardState`):", "Implements unified state container with dictionary-like access, designed for seamless LangGraph agent migration.", pt_size=10.0)
    add_bullet_item(tf_uip, "Singleton Agent Caching:", "Models stay preloaded in GPU VRAM (`_seg_agent`, `_clf_agent`), eliminating reload latency across inferences.", pt_size=10.0)
    add_bullet_item(tf_uip, "Dual-Resolution Pipeline:", "Feeds 224x224 tensor to classification agent and 256x256 tensor to segmentation agent from a single raw upload.", pt_size=10.0)
    add_bullet_item(tf_uip, "Safety & Exception Logging:", "Catches corrupted slices, low-contrast inputs, or extreme aspect ratios with graceful clinician alerts.", pt_size=10.0)

    # =========================================================================
    # SLIDE 13: WORK PLAN & MILESTONE TRACKING
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_header(s13, "SECTION 5: WORK PLAN", "Project Milestone Status & Completion Matrix",
                     "Audit of completed technical milestones against semester objectives", slide_num=13)

    # Table of Milestones
    add_card(s13, Inches(0.8), Inches(1.7), Inches(11.73), Inches(5.1), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

    tbl_s13 = s13.shapes.add_table(9, 4, Inches(1.05), Inches(1.85), Inches(11.23), Inches(4.7))
    t13 = tbl_s13.table
    t13.columns[0].width = Inches(2.23)
    t13.columns[1].width = Inches(4.8)
    t13.columns[2].width = Inches(2.2)
    t13.columns[3].width = Inches(2.0)

    milestone_headers = ["Phase", "Technical Scope & Deliverables", "Target Technology", "Current Status"]
    for idx, h in enumerate(milestone_headers):
        cell = t13.cell(0, idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY_NAVY
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    milestones = [
        ("Phase 0: Setup", "Repository architecture, virtual env, CUDA 12.4 configuration", "PyTorch / Conda", "Completed"),
        ("Phase 1: Preprocessing", "Validation, CLAHE normalization, dual-size transforms, dataloaders", "OpenCV / Torchvision", "Completed"),
        ("Phase 2: Segmentation", "Benchmark 6 architectures; train ResNet-34 U-Net++ (87.78% Dice)", "PyTorch / CUDA", "Completed"),
        ("Phase 3: Classification", "Transfer learning with EfficientNet-B0 (99.00% Test Accuracy)", "timm / PyTorch", "Completed"),
        ("Phase 4: Feature Extraction", "Morphological biomarkers (area, perimeter, centroid, intensity)", "scikit-image / NumPy", "Active (70%)"),
        ("Phase 5: Evidence (RAG)", "Vector store construction, guideline embeddings & retrieval", "ChromaDB / all-MiniLM", "Scheduled (Nov)"),
        ("Phase 6: Chief Synthesis", "Structured clinical radiology reporting agent with guardrails", "Gemini 2.0 Flash", "Scheduled (Nov/Dec)"),
        ("Phase 7: Full Integration", "LangGraph multi-agent loop, latency profiling, final report", "LangGraph / Streamlit", "Scheduled (Dec)"),
    ]

    for row_idx, r_data in enumerate(milestones):
        bg = BG_LIGHT if row_idx % 2 == 0 else WHITE
        for col_idx, val in enumerate(r_data):
            cell = t13.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_DARK
            if col_idx == 0:
                p.font.bold = True
            if col_idx == 3:
                p.font.bold = True
                p.font.color.rgb = PRIMARY_NAVY if val == "Completed" else SECONDARY_NAVY
                p.alignment = PP_ALIGN.CENTER
            elif col_idx != 1:
                p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 14: FUTURE WORK & TIMELINE (GANTT BREAKDOWN)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_slide_header(s14, "SECTION 5: FUTURE WORK & TIMELINE", "Semester Timeline & Architecture Roadmap",
                     "Structured schedule for the remainder of B.Tech Project Work (Odd Semester)", slide_num=14)

    roadmap = [
        ("WEEKS 8–9 (LATE OCT)", "Feature Extraction & Morphometrics",
         "• Finalize 2D contour geometry calculations (lesion area, perimeter, circularity, centroid).\n"
         "• Extract tumor-to-brain volume fraction and spatial localization relative to anatomical midline.\n"
         "• Deliverable: Fully verified `feature_extraction_agent.py` integrated into pipeline state."),

        ("WEEKS 10–11 (MID NOV)", "RAG Medical Knowledge Base",
         "• Ingest WHO CNS5 neuro-oncology diagnostic criteria & clinical radiology practice guidelines into ChromaDB.\n"
         "• Implement `all-MiniLM-L6-v2` dense vector semantic search for guideline matching.\n"
         "• Deliverable: Autonomous `evidence_agent.py` returning top-3 grounded clinical references."),

        ("WEEKS 12–13 (LATE NOV)", "Chief Synthesis Agent (Gemini LLM)",
         "• Integrate Gemini 2.0 Flash via API with structured clinical prompt templates.\n"
         "• Synthesize classifications, segmentation masks, and guideline evidence into formal draft radiology reports.\n"
         "• Deliverable: Automated structured diagnostic report generation with academic safety disclaimers."),

        ("WEEKS 14–15 (DECEMBER)", "System Integration & Final Evaluation",
         "• Migrate pipeline to LangGraph autonomous multi-agent state graph with human-in-the-loop review.\n"
         "• Comprehensive latency profiling, stress testing on challenging edge cases, and UI polishing.\n"
         "• Deliverable: End-Semester project presentation and comprehensive B.Tech project thesis."),
    ]

    card_w14 = Inches(2.78)
    card_h14 = Inches(5.1)
    lefts_14 = [Inches(0.8), Inches(3.78), Inches(6.76), Inches(9.74)]

    for (period, task_title, details), cl in zip(roadmap, lefts_14):
        add_card(s14, cl, Inches(1.7), card_w14, card_h14, bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        # Header Pill (Static Navy)
        h_pill = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, cl, Inches(1.7), card_w14, Inches(0.42))
        h_pill.fill.solid()
        h_pill.fill.fore_color.rgb = PRIMARY_NAVY
        h_pill.line.fill.background()
        tf_hp = h_pill.text_frame
        p_hp = tf_hp.paragraphs[0]
        p_hp.text = period
        p_hp.font.name = "Calibri"
        p_hp.font.size = Pt(10)
        p_hp.font.bold = True
        p_hp.font.color.rgb = WHITE
        p_hp.alignment = PP_ALIGN.CENTER

        tb_rm = s14.shapes.add_textbox(cl + Inches(0.2), Inches(2.22), card_w14 - Inches(0.4), card_h14 - Inches(0.65))
        tf_rm = tb_rm.text_frame
        tf_rm.word_wrap = True

        pt = tf_rm.paragraphs[0]
        pt.text = task_title
        pt.font.name = "Calibri"
        pt.font.size = Pt(11.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY
        pt.space_after = Pt(8)

        pb = tf_rm.add_paragraph()
        pb.text = details
        pb.font.name = "Calibri"
        pb.font.size = Pt(9.5)
        pb.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 15: SUMMARY & CONCLUSION
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_slide_header(s15, "SUMMARY & CONCLUSION", "Mid-Semester Milestones & Key Scientific Takeaways",
                     "Recap of verified experimental results and readiness for final multi-agent synthesis", slide_num=15)

    summary_cards = [
        ("1. High-Accuracy Classification Verified",
         "Successfully achieved 99.00% test accuracy and 0.9899 Macro-F1 with EfficientNet-B0 across 1,000 unseen BRISC 2025 brain MRI scans with <15 ms latency, proving transfer learning efficacy for 4-class intracranial neoplasm triage."),

        ("2. Empirical Proof of Nested Skips in Segmentation",
         "Completed rigorous benchmark of 6 segmentation paradigms across 860 test scans. Pretrained ResNet-34 U-Net++ established new performance benchmark at 87.78% Dice Score (81.11% Mean IoU), outperforming standard U-Net baseline."),

        ("3. Interactive Clinical Decision-Support Ready",
         "Built a functional Streamlit interface (`ui/app.py`) providing immediate real-time tumor contour overlays, multi-class probability graphs, and key quantitative diagnostic metrics for radiologist inspection."),

        ("4. Strong Technical Foundation for Agentic Reasoning",
         "The modular architecture (`pipeline.py`, `MedBoardState`) is fully primed for upcoming RAG vector store integration (ChromaDB) and generative clinical reporting (Gemini 2.0 Flash) in the second half of the semester."),
    ]

    top_s15 = 1.7
    for num_t, desc in summary_cards:
        add_card(s15, Inches(0.8), Inches(top_s15), Inches(11.73), Inches(1.15), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

        # Left subtle accent bar
        l_bar = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(top_s15), Inches(0.08), Inches(1.15))
        l_bar.fill.solid()
        l_bar.fill.fore_color.rgb = PRIMARY_NAVY
        l_bar.line.fill.background()

        tb = s15.shapes.add_textbox(Inches(1.05), Inches(top_s15 + 0.1), Inches(11.3), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = num_t
        pt.font.name = "Calibri"
        pt.font.size = Pt(11.5)
        pt.font.bold = True
        pt.font.color.rgb = PRIMARY_NAVY
        pt.space_after = Pt(2)

        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.name = "Calibri"
        pd.font.size = Pt(10)
        pd.font.color.rgb = TEXT_DARK

        top_s15 += 1.3

    # =========================================================================
    # SLIDE 16: Q&A / ACKNOWLEDGMENTS
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)

    # Full Background (Static Navy)
    bg_full = s16.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_WIDTH, SLIDE_HEIGHT)
    bg_full.fill.solid()
    bg_full.fill.fore_color.rgb = PRIMARY_NAVY
    bg_full.line.fill.background()

    # Center Box
    tb_end = s16.shapes.add_textbox(Inches(1.5), Inches(1.2), Inches(10.33), Inches(1.9))
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
    p_qa.text = "Questions, Feedback & Discussion"
    p_qa.font.name = "Calibri"
    p_qa.font.size = Pt(19)
    p_qa.font.color.rgb = RGBColor(191, 219, 254)
    p_qa.alignment = PP_ALIGN.CENTER

    # Bottom Details Card
    add_card(s16, Inches(1.8), Inches(3.5), Inches(9.73), Inches(3.3), bg_color=BG_LIGHT, border_color=BORDER_SUBTLE)

    tb_inf = s16.shapes.add_textbox(Inches(2.1), Inches(3.7), Inches(9.13), Inches(2.9))
    tf_inf = tb_inf.text_frame
    tf_inf.word_wrap = True

    pi_t = tf_inf.paragraphs[0]
    pi_t.text = "MedBoard: Multi-Agent Brain MRI Analysis & Automated Reporting System"
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

    # Save presentation
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    print(f"✅ Professional Academic Mid-Sem PPT generated successfully at: {output_path}")


if __name__ == "__main__":
    out_p = "docs/MedBoard_MidSem_Evaluation.pptx"
    if len(sys.argv) > 1:
        out_p = sys.argv[1]
    build_midsem_presentation(out_p)
