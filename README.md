# MedBoard

**Multi-Agent Brain MRI Analysis & Preliminary Reporting System**

> ⚠️ **Academic Disclaimer**: MedBoard is a research prototype developed as a final-year academic project. It is **NOT** a clinical diagnostic tool and must **NOT** be used for medical decision-making without qualified medical review.

---

## Overview

MedBoard is an end-to-end AI pipeline that takes a brain MRI image and produces a structured preliminary analysis report by orchestrating multiple specialized agents:

```
MRI Input → Preprocessing → Segmentation → Classification → Feature Extraction → Evidence Retrieval → Report Generation
```

## Dataset

**BRISC 2025** — Brain tumor Image Segmentation & Classification
- 6,000 T1-weighted MRI slices (2D), 5,000 train / 1,000 test
- 4 classes: Glioma · Meningioma · Pituitary Tumor · No Tumor
- Paired binary segmentation masks
- Source: arXiv:2506.14318, Scientific Data (Nature Portfolio)

## Technology Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch + timm (EfficientNet-B0) |
| Segmentation | Custom U-Net |
| Agent Orchestration | LangGraph |
| LLM Synthesis | Gemini 2.0 Flash |
| RAG / Evidence | sentence-transformers + ChromaDB |
| UI | Streamlit |
| Compute | RTX 2050 4GB (local) / Google Colab (backup) |

## Project Structure

```
Major/
├── agents/                    # LangGraph agent implementations
│   ├── segmentation_agent.py
│   ├── classification_agent.py
│   ├── feature_extraction_agent.py
│   ├── evidence_agent.py
│   └── synthesis_agent.py
├── modules/                   # Core ML modules (non-agent logic)
│   ├── preprocessing.py
│   ├── unet.py
│   └── feature_extractor.py
├── rag/                       # RAG components
│   ├── knowledge_base/        # JSON medical knowledge entries
│   ├── vector_store/          # ChromaDB index
│   └── embedder.py
├── ui/                        # Streamlit app
│   └── app.py
├── data/                      # Dataset (not committed to git)
│   ├── raw/                   # BRISC 2025 original files
│   ├── processed/             # Preprocessed tensors
│   └── samples/               # Test images for dev
├── models/                    # Saved model weights (not committed)
│   ├── segmentation/
│   └── classification/
├── notebooks/                 # Training & exploration notebooks
├── tests/                     # Unit and integration tests
├── configs/
│   └── config.yaml            # Central configuration
├── outputs/
│   └── reports/               # Generated reports
├── pipeline.py                # End-to-end pipeline entry point
├── requirements.txt
├── requirements_colab.txt
└── .env.example
```

## Setup

### 1. Clone and set up environment

```bash
cd d:/Projects/Major
python -m venv venv
venv\Scripts\activate          # Windows
```

### 2. Install PyTorch with CUDA (Python 3.13 — requires cu124)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 5. Download BRISC 2025 dataset

```bash
# Place kaggle.json at C:\Users\<you>\.kaggle\kaggle.json first
kaggle datasets download -d <brisc2025-dataset-id> -p data/raw --unzip
```

### 6. Run the app

```bash
streamlit run ui/app.py
```

## Development Phases

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Project setup & structure | ✅ Complete |
| Phase 1 | Preprocessing module | ⏳ Pending |
| Phase 2 | Segmentation Agent (U-Net) | ⏳ Pending |
| Phase 3 | Classification Agent (EfficientNet-B0) | ⏳ Pending |
| Phase 4 | Feature Extraction module | ⏳ Pending |
| Phase 5 | Evidence / RAG Agent | ⏳ Pending |
| Phase 6 | Chief Synthesis Agent (Gemini) | ⏳ Pending |
| Phase 7 | Streamlit UI | ⏳ Pending |
| Phase 8 | End-to-end integration & testing | ⏳ Pending |

## Academic References

- BRISC 2025: arXiv:2506.14318
- U-Net: Ronneberger et al., 2015
- EfficientNet: Tan & Le, 2019
- LangGraph: LangChain framework
- ChromaDB: Chroma vector database
