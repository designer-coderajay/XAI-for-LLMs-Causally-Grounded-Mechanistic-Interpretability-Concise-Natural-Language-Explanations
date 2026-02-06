# Explainable AI for LLMs via Mechanistic Interpretability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## MSc Thesis Project

**Author:** Ajay Mahale  
**Institution:** Hochschule Trier  
**Supervisor:** Prof. Dr. Haffner  
**Date:** May 2026

---

## 📋 Overview

This repository contains the experimental code for my MSc thesis on generating **causally faithful natural language explanations** from mechanistic circuit analysis of transformer language models.

### Key Contributions

1. **Circuit → LLM → NL Pipeline**: First systematic use of LLMs to generate natural language explanations from mechanistic circuit analysis
2. **ERASER Adaptation**: Novel application of ERASER faithfulness metrics to mechanistic interpretability
3. **Template vs Learned Comparison**: Quantitative evaluation showing learned explanations achieve +63% quality improvement
4. **Failure Analysis**: Systematic taxonomy of when/why explanations diverge from mechanisms

### Results Summary

| Metric | Value |
|--------|-------|
| IOI Circuit Coverage | 61.4% (6 heads) |
| Sufficiency | 100.0% |
| Comprehensiveness | 22.0% |
| Local Faithfulness Score | 36.0% |
| Beat Attention Baseline | +75% |
| Learned vs Template | +63% |

---

## ⚠️ Important Notes

### Scope Limitation
**All results are restricted to Indirect Object Identification (IOI) tasks and should not be interpreted as general LLM explanation performance.** This work validates the methodology on a well-characterized circuit; generalization to other tasks requires further investigation.

### Environment Warnings
CUDA / Triton / TensorFlow warnings may appear on CPU systems and can be safely ignored. All experiments were run on CPU unless otherwise stated.

### Example Similarity
Mechanistic patterns are stable across IOI prompts by design; variation appears primarily in model confidence and comprehensiveness scores. This stability demonstrates the robustness of the identified circuit, not a limitation of the analysis.

---

## 📁 Repository Structure
```
thesis/
├── notebooks/           # Jupyter notebooks (01-10)
│   ├── 01_setup_test.ipynb
│   ├── 02_ioi_reproduction.ipynb
│   ├── 03_nl_explanation_generator.ipynb
│   ├── 04_baselines_comparison.ipynb
│   ├── 05_expanded_evaluation.ipynb
│   ├── 06_failure_analysis_main.ipynb
│   ├── 07_esnli_format_study.ipynb
│   ├── 08_learned_nl_generator.ipynb      # Private
│   ├── 09_template_vs_learned.ipynb       # Private
│   └── 10_final_evaluation.ipynb
├── results/             # Experiment results (.pkl files)
├── figures/             # Generated plots and figures
├── src/                 # Source code
│   ├── demo_pipeline.py # ← MAIN ENTRY POINT
│   └── 00_reproducibility_config.py
├── data/                # Data files
├── docs/                # Documentation
├── thesis_writing/      # Thesis chapters
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/thesis-mechanistic-interp.git
cd thesis-mechanistic-interp
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Demo (Main Entry Point)
```bash
python src/demo_pipeline.py
```

### 4. Run in Google Colab
Open any notebook in `notebooks/` folder with Google Colab.

---

## 📓 Notebooks Overview

| Notebook | Purpose | GPU Required |
|----------|---------|--------------|
| 01 | Environment setup & verification | No |
| 02 | IOI circuit reproduction | Yes |
| 03 | Template NL explanation generator | Yes |
| 04 | Baseline comparison (preliminary) | Yes |
| 05 | Full ERASER evaluation (canonical) | Yes |
| 06 | Failure analysis (RQ3) | Yes |
| 07 | e-SNLI format study | No |
| 08 | Learned NL generator (novel) | Yes |
| 09 | Template vs learned comparison | Yes |
| 10 | Final evaluation & summary | No |

**Note:** Notebooks 08-09 use the Anthropic API and are kept private.

---

## 🎯 Demo

Run the interactive demo (main entry point):
```bash
python src/demo_pipeline.py
```

Example output:
```
📝 INPUT:
   When Mary and John went to the store, John gave a drink to

🎯 PREDICTION:
   "Mary" (Model Confidence: 85.3%)

🔬 MECHANISTIC EVIDENCE:
   L9H9  | Name Mover (Primary)  | 17.4% (Avg) | 66.5% → Mary

✅ TRUST ASSESSMENT:
   Trust Level: MEDIUM
   (High confidence but low comprehensiveness suggests backup circuits)
```

---

## 📊 Metric Definitions

| Metric | Definition |
|--------|------------|
| **Sufficiency** | Prediction preserved when using ONLY the cited heads |
| **Comprehensiveness** | Prediction reduction when ABLATING the cited heads |
| **Local Faithfulness Score** | ERASER-style proxy: harmonic mean of Sufficiency and Comprehensiveness |
| **Global Importance** | Dataset-level average from ablation experiments (following IOI methodology) |
| **Attention Weight** | Instance-level correlation (NOT causation) |

---

## ⚠️ Limitations

- **Model**: Validated only on GPT-2 Small (124M parameters)
- **Task**: Only works for Indirect Object Identification (IOI) tasks
- **Generalization**: Results may not generalize to larger models or other tasks
- **Comprehensiveness**: 22% average indicates significant distributed computation

---

## 📚 References

Key papers this work builds on:

1. Wang et al. (2022) - "Interpretability in the Wild: IOI Circuit"
2. DeYoung et al. (2020) - "ERASER Benchmark"
3. Elhage et al. (2021) - "Mathematical Framework for Transformer Circuits"
4. Bills et al. (2023) - "Language Models Can Explain Neurons"
5. Jain & Wallace (2019) - "Attention is Not Explanation"

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Prof. Dr. Haffner (Supervisor)
- Hochschule Trier
- TransformerLens library by Neel Nanda
- Anthropic for Claude API access
