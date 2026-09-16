"""
MedBoard — ui/app.py
Interactive Multi-Agent Brain MRI Analysis & Reporting Dashboard

Runs the full pipeline (Preprocessing, U-Net Segmentation, and EfficientNet-B0 Classification)
with an intuitive, modern clinical research interface.

Run locally:
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import MedBoardState, run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedBoard | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-glioma { color: #DC2626; font-weight: bold; }
    .badge-meningioma { color: #D97706; font-weight: bold; }
    .badge-pituitary { color: #7C3AED; font-weight: bold; }
    .badge-no_tumor { color: #059669; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar: System Hardware, Model Info, Sample Picker
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.title("MedBoard Control")
    st.caption("v0.1.0 • Academic Research Prototype")

    # Hardware Info
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Fallback)"
    st.success(f"**Compute**: {gpu_name}")

    st.markdown("---")
    st.subheader("🤖 Active Models")
    st.markdown(
        """
        - **U-Net Segmentation**
          - 7.85M parameters • 256×256
          - Test Dice: **0.8706** (Excellent)
        - **EfficientNet-B0 Classifier**
          - 4.01M parameters • 224×224
          - 4-Class: Glioma, Meningioma, Pituitary, Normal
        """
    )

    st.markdown("---")
    st.subheader("🧪 Quick Test Samples")
    st.caption("Select a sample scan from the BRISC test set:")

    def find_sample_file(class_name: str) -> Optional[str]:
        cls_folder = PROJECT_ROOT / "data" / "raw" / "classification_task" / "test" / class_name
        if cls_folder.exists():
            for file in sorted(cls_folder.glob("*.jpg")):
                return str(file.relative_to(PROJECT_ROOT))
        return None

    sample_dict = {
        "None (Use Upload)": None,
        "Glioma Sample": find_sample_file("glioma"),
        "Meningioma Sample": find_sample_file("meningioma"),
        "Pituitary Sample": find_sample_file("pituitary"),
        "No Tumor (Healthy)": find_sample_file("no_tumor"),
    }

    selected_sample = st.selectbox("Load Demo Case", list(sample_dict.keys()), index=0)

    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer**: MedBoard is an academic prototype for educational and research exploration only. "
        "It is not validated for clinical diagnosis."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_mask_overlay(
    original_image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.45,
) -> np.ndarray:
    """Create a transparent color overlay of the tumor mask onto the MRI scan."""
    img_np = np.array(original_image.convert("RGB"))
    h, w = img_np.shape[:2]

    # Resize mask to original image resolution
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = img_np.copy()
    tumor_indices = mask_resized > 0

    if np.any(tumor_indices):
        colored_mask = np.zeros_like(img_np)
        colored_mask[tumor_indices] = color

        # Blend image and colored mask
        cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)

        # Draw smooth contour line around tumor boundary
        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Main Interface
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🧠 MedBoard Diagnostic Workspace</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Agent MRI Localization & Tumor Classification System</div>',
    unsafe_allow_html=True,
)

col_input, col_output = st.columns([1, 2], gap="medium")

image_path_to_process: Optional[str] = None
uploaded_image_display: Optional[Image.Image] = None

with col_input:
    st.subheader("1. Input Brain MRI")

    # Handle sample selection or manual upload
    if selected_sample != "None (Use Upload)":
        rel_path = sample_dict.get(selected_sample)
        if rel_path:
            sample_path = PROJECT_ROOT / rel_path
            if sample_path.exists():
                image_path_to_process = str(sample_path)
                uploaded_image_display = Image.open(image_path_to_process)
                st.info(f"Loaded: **{selected_sample}** (`{sample_path.name}`)")
            else:
                st.error(f"Sample file not found at: {sample_path}")
        else:
            st.warning(f"No sample scans found in test folder for {selected_sample}.")

    uploaded_file = st.file_uploader(
        "Or Upload New MRI Scan (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an axial T1-weighted brain MRI slice.",
    )

    if uploaded_file is not None:
        # Save temporary file for pipeline processing
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_file.read())
        tfile.close()
        image_path_to_process = tfile.name
        uploaded_image_display = Image.open(image_path_to_process)

    if uploaded_image_display is not None:
        st.image(
            uploaded_image_display,
            caption=f"Source MRI ({uploaded_image_display.width} × {uploaded_image_display.height})",
            use_container_width=True,
        )
        run_analysis = st.button("🚀 Analyze Scan", type="primary", use_container_width=True)
    else:
        st.info("👈 Upload an MRI scan or choose a Quick Demo Case to begin.")
        run_analysis = False


with col_output:
    st.subheader("2. Multi-Agent Analysis")

    if not image_path_to_process or not run_analysis:
        st.markdown(
            """
            <div style="padding: 24px; background: #F8FAFC; border-radius: 8px; border: 1px dashed #CBD5E1;">
                <h4 style="color: #475569; margin-top: 0;">System Ready</h4>
                <p style="color: #64748B;">Once you click <b>Analyze Scan</b>, the multi-agent pipeline will execute:</p>
                <ol style="color: #64748B; margin-bottom: 0;">
                    <li><b>Preprocessing Node</b>: Image normalization and tensor alignment</li>
                    <li><b>Segmentation Agent</b>: U-Net pixel-level tumor boundary detection</li>
                    <li><b>Classification Agent</b>: EfficientNet-B0 probability distribution</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("⚙️ Initializing Preprocessing Node...")
        progress_bar.progress(20)
        t_start = time.time()

        status_text.text("🔍 Running U-Net Segmentation Agent (GPU)...")
        progress_bar.progress(50)

        # Run pipeline
        state = run_pipeline(image_path_to_process)

        status_text.text("🏷️ Running EfficientNet-B0 Classification Agent...")
        progress_bar.progress(85)
        time.sleep(0.1)

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        t_elapsed = time.time() - t_start

        # Check for errors
        if state.errors:
            for err in state.errors:
                st.error(err)

        # ── KPI Summary Cards ────────────────────────────────────────────────
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            cls_name = state.predicted_class or "N/A"
            cls_color = {
                "glioma": "#DC2626",
                "meningioma": "#D97706",
                "pituitary": "#7C3AED",
                "no_tumor": "#059669",
            }.get(cls_name, "#1E293B")

            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Diagnosis</div>
                    <div class="kpi-value" style="color: {cls_color};">{cls_name.replace('_', ' ').title()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi_col2:
            conf_val = f"{state.classification_confidence * 100:.1f}%"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Confidence</div>
                    <div class="kpi-value" style="color: #2563EB;">{conf_val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi_col3:
            tumor_flag = "DETECTED" if state.has_tumor else "NONE"
            flag_color = "#DC2626" if state.has_tumor else "#059669"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Tumor Region</div>
                    <div class="kpi-value" style="color: {flag_color};">{tumor_flag}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi_col4:
            area_str = f"{state.segmentation_confidence:.2f}%"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Tumor Area</div>
                    <div class="kpi-value" style="color: #0F172A;">{area_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Detailed Result Tabs ─────────────────────────────────────────────
        tab_seg, tab_clf, tab_report, tab_trace = st.tabs([
            "🗺️ Segmentation Localization",
            "📊 Classification Breakdown",
            "📄 Preliminary Report",
            "⚡ Agent Pipeline Trace",
        ])

        # TAB 1: SEGMENTATION
        with tab_seg:
            st.markdown("##### 3-Panel Tumor Localization")
            p_col1, p_col2, p_col3 = st.columns(3)

            with p_col1:
                st.image(uploaded_image_display, caption="1. Original MRI", use_container_width=True)

            with p_col2:
                if state.tumor_mask is not None:
                    # Binary mask display
                    mask_vis = (state.tumor_mask * 255).astype(np.uint8)
                    st.image(mask_vis, caption="2. Predicted Mask (256×256)", use_container_width=True)
                else:
                    st.info("No mask produced.")

            with p_col3:
                if state.tumor_mask is not None:
                    overlay_img = create_mask_overlay(uploaded_image_display, state.tumor_mask)
                    st.image(overlay_img, caption="3. Visual Overlay (Boundary)", use_container_width=True)
                else:
                    st.info("No overlay available.")

            if state.tumor_mask is not None:
                tumor_pixels = int(np.sum(state.tumor_mask))
                total_pixels = state.tumor_mask.size
                st.caption(
                    f"**Mask Quantification**: Tumor covers **{tumor_pixels:,}** of {total_pixels:,} pixels "
                    f"({state.segmentation_confidence:.2f}% of slice area)."
                )

        # TAB 2: CLASSIFICATION
        with tab_clf:
            st.markdown("##### Model Probability Distribution")
            if state.class_probabilities:
                probs_df = pd.DataFrame([
                    {
                        "Tumor Type": k.replace("_", " ").title(),
                        "Probability (%)": round(v * 100, 2),
                    }
                    for k, v in state.class_probabilities.items()
                ]).sort_values("Probability (%)", ascending=False)

                st.bar_chart(probs_df.set_index("Tumor Type"), color="#2563EB")
                st.dataframe(probs_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("##### Clinical Reference")
            tumor_descriptions = {
                "glioma": "Gliomas originate from glial cells in the brain parenchyma and represent one of the most common primary brain neoplasms. Requires prompt neuro-oncologic evaluation.",
                "meningioma": "Meningiomas arise from the arachnoid cells of the meninges. Most meningiomas are benign and slow-growing, though compressive symptoms can occur.",
                "pituitary": "Pituitary tumors typically arise in the sella turcica and may affect endocrine balance or cause visual field deficits due to optic chiasm compression.",
                "no_tumor": "No focal pathological mass or hyperintense lesion detected consistent with neoplastic intracranial processes.",
            }
            desc = tumor_descriptions.get(state.predicted_class, "")
            st.write(desc)

        # TAB 3: PRELIMINARY REPORT
        with tab_report:
            st.markdown("##### Automated Preliminary Radiology Summary")
            report_text = f"""
### MedBoard AI Preliminary Impression
**Date / Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Study Type**: Brain MRI (Axial T1-weighted)  
**Examined Scan**: `{Path(image_path_to_process).name}`  

---

#### 1. Findings
- **Tumor Region**: {'Focal abnormal tissue detected' if state.has_tumor else 'No focal mass detected'}.
- **Relative Area**: {state.segmentation_confidence:.2f}% of brain slice.
- **Top Predicted Classification**: **{state.predicted_class.replace('_', ' ').upper()}**
- **Classification Confidence**: {state.classification_confidence * 100:.1f}%

#### 2. Class Distribution
"""
            for cls, prob in state.class_probabilities.items():
                report_text += f"- **{cls.replace('_', ' ').title()}**: {prob * 100:.1f}%\n"

            report_text += """
---
> ⚠️ **CONFIDENTIAL & RESEARCH PURPOSES ONLY**  
> This preliminary summary was automatically synthesized by the MedBoard multi-agent system.
> It has not been signed off by a certified medical radiologist. Do not use for definitive patient management.
"""
            st.markdown(report_text)
            st.download_button(
                "📥 Download Summary (.txt)",
                report_text,
                file_name="medboard_summary.txt",
                mime="text/plain",
            )

        # TAB 4: TRACE
        with tab_trace:
            st.markdown("##### Pipeline Node Execution Log")
            st.json({
                "Execution Time (s)": round(t_elapsed, 3),
                "Hardware": gpu_name,
                "CUDA Enabled": cuda_avail,
                "Image Source": Path(image_path_to_process).name,
                "Nodes Executed": [
                    "preprocessing_node",
                    "segmentation_agent_unet",
                    "classification_agent_efficientnet_b0",
                ],
                "Segmentation Mask Shape": str(state.tumor_mask.shape) if state.tumor_mask is not None else None,
                "Predicted Class": state.predicted_class,
                "Confidence Score": state.classification_confidence,
                "Pipeline Errors": state.errors,
            })
