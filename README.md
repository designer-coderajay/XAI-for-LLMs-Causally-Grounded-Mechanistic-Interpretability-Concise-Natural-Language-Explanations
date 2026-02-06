# Causally Grounded Mechanistic Interpretability and Concise Natural-Language Explanations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

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

## Novelty Statement

To our knowledge, this is the first work to:
1. Systematically translate mechanistic circuit analysis into natural language explanations
2. Apply ERASER faithfulness metrics to mechanistic interpretability
3. Provide quantitative comparison of template-based vs LLM-generated explanations
4. Analyze failure modes where explanations diverge from underlying mechanisms

   
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

## 🎯 Demo Output

Run the interactive demo:
```bash
python src/demo_pipeline.py
```

**Example Output:**
```
======================================================================
MECHANISTIC EXPLANATION
======================================================================

📝 INPUT:
   When Diana and Steve went to the museum, Steve showed a painting to

🎯 PREDICTION:
   "Diana" (Model Confidence: 71.6%)

🔬 MECHANISTIC EVIDENCE:
----------------------------------------------------------------------
   Head     | Role                   | Global Imp.  | Attention   
   -------- | ---------------------- | ------------ | ------------
   L9H9     | Name Mover (Primary)   |  17.4% (Avg) |  83.9% → Diana
   L8H10    | S-Inhibition           |  12.3% (Avg) |   8.1% → Diana
   L7H3     | Name Mover (Secondary) |  10.3% (Avg) |   0.8% → Diana

📊 FAITHFULNESS METRICS (ERASER-style):
   • Sufficiency:        100.0%
   • Comprehensiveness:  22.4%
   • Local Faithfulness: 44.4%

💬 EXPLANATION:
   The model predicts 'Diana' because the Name Mover head L9H9 
   attends to 'Diana' with 83.9% attention, copying it to output.
   The S-Inhibition head suppresses 'Steve' (the giver).
```

> ⚠️ **Note:** Low comprehensiveness (~22%) indicates backup circuits contribute significantly. This is a known limitation.
````



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

### Core Methodology
- Wang, K. et al. (2022). "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small." *ICLR 2023*.
- Nanda, N. et al. (2022). "TransformerLens." GitHub. https://github.com/TransformerLensOrg/TransformerLens
- Nanda, N. & Lieberum, T. (2022). "A Comprehensive Mechanistic Interpretability Explainer & Glossary." https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J
- Elhage, N. et al. (2021). "A Mathematical Framework for Transformer Circuits." *Anthropic*.
- Conmy, A. et al. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability." *NeurIPS 2023*.

### Evaluation & Faithfulness
- DeYoung, J. et al. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models." *ACL 2020*.
- Jain, S. & Wallace, B. (2019). "Attention is not Explanation." *NAACL 2019*.

### LLM-Based Explanations
- Bills, S. et al. (2023). "Language Models Can Explain Neurons in Language Models." *OpenAI*.
- Anthropic (2024). Claude API. https://www.anthropic.com

### Causal Interpretability
- Geiger, A. et al. (2021). "Causal Abstractions of Neural Networks." *NeurIPS 2021*.
- Meng, K. et al. (2022). "Locating and Editing Factual Associations in GPT." *NeurIPS 2022*.
- Goldowsky-Dill, N. et al. (2023). "Localizing Model Behavior with Path Patching." *arXiv*.

### Foundational
- Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.
- Radford, A. et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI*.
- Olah, C. et al. (2020). "Zoom In: An Introduction to Circuits." *Distill*.
- Olsson, C. et al. (2022). "In-context Learning and Induction Heads." *Anthropic*.

---

## 📄 License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — View, study, and cite with attribution. Commercial use and derivatives require written permission.

---

## 🙏 Acknowledgments

- Prof. Dr. Haffner (Supervisor)
- Hochschule Trier
- TransformerLens library by Neel Nanda
- Anthropic for Claude API access
