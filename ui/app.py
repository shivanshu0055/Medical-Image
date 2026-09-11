"""
MedBoard — Streamlit UI (skeleton)
Full implementation comes in Phase 7.
This file is a verified shell that proves the app starts correctly.
"""

import streamlit as st

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedBoard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Header ──────────────────────────────────────────────────
st.title("🧠 MedBoard")
st.subheader("Multi-Agent Brain MRI Analysis & Preliminary Reporting System")
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("About MedBoard")
    st.info(
        "MedBoard is an academic AI prototype that analyzes brain MRI images "
        "using a multi-agent pipeline (LangGraph + Gemini)."
    )
    st.warning(
        "⚠️ **Disclaimer**: This is a research prototype. "
        "Results must NOT be used for clinical diagnosis."
    )
    st.markdown("---")
    st.markdown("**Pipeline Stages**")
    stages = [
        "1. Preprocessing",
        "2. Segmentation Agent",
        "3. Classification Agent",
        "4. Feature Extraction",
        "5. Evidence / RAG Agent",
        "6. Chief Synthesis Agent",
    ]
    for stage in stages:
        st.markdown(f"- {stage}")

# ─── Main Content ─────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 Upload MRI")
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image",
        type=["jpg", "jpeg", "png"],
        help="Upload a 2D T1-weighted brain MRI slice.",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)
        st.success("Image uploaded successfully!")
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    else:
        st.info("Please upload an MRI image to begin.")
        run_btn = False

with col2:
    st.header("📋 Analysis Results")

    if not uploaded_file:
        st.markdown(
            """
            Results will appear here after you upload an MRI and click **Run Analysis**.

            The pipeline will produce:
            - **Segmentation** — tumor region overlay
            - **Classification** — tumor type & confidence
            - **Features** — quantitative tumor measurements
            - **Evidence** — supporting medical references
            - **Report** — structured preliminary report
            """
        )
    elif run_btn:
        # Placeholder — real pipeline wired up in Phase 8
        with st.spinner("Running pipeline... (placeholder)"):
            st.info(
                "🔧 Pipeline not yet implemented. "
                "This placeholder will be replaced in Phase 7–8."
            )

        # Result tabs (empty for now, wired up in Phase 7)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🗺️ Segmentation", "🏷️ Classification", "📊 Features", "📚 Evidence", "📄 Report"]
        )
        with tab1:
            st.info("Segmentation results will appear here.")
        with tab2:
            st.info("Classification results will appear here.")
        with tab3:
            st.info("Feature extraction results will appear here.")
        with tab4:
            st.info("Evidence retrieval results will appear here.")
        with tab5:
            st.info("Preliminary report will appear here.")
