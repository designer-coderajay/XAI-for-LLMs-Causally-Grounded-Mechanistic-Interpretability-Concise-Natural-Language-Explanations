# Causally Grounded Mechanistic Interpretability and Concise Natural-Language Explanations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.09988-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.09988)
## MSc Thesis Project

Sc Thesis: Causally Grounded Mechanistic Interpretability and Concise Natural-Language Explanations
Author: Ajay Pravin Mahale
University Mail ID: jymh0144@hochschule-trier.de
Personal Mail ID: Mahale.ajay01@gmail.com
Institution: Hochschule Trier
Supervisor: Prof. Dr. Ernst Georg Haffner

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
│   ├── 08_learned_nl_generator.ipynb      
│   ├── 09_template_vs_learned.ipynb      
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
MECHANISTIC INTERPRETABILITY DEMO
Explainable AI for LLMs via Circuit Analysis
======================================================================

╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  DOMAIN: Indirect Object Identification (IOI) Tasks Only     ║
║  Enter sentences like:                                           ║
║  "When [Name1] and [Name2] went to X, [Name2] gave Y to"         ║
╚══════════════════════════════════════════════════════════════════╝

📋 EXAMPLE PROMPTS (10 available):
    1. When Mary and John went to the store, John gave a dr...
    2. When Alice and Bob went to the park, Bob handed a fl...
    3. When Sarah and Tom went to the library, Tom showed a...
    4. When Emma and James went to the cafe, James offered ...
    5. When Lisa and David went to the beach, David threw a...
    6. When Sophie and Daniel went to the party, Daniel gav...
    7. When Rachel and Chris went to the office, Chris sent...
    8. When Laura and Kevin went to the restaurant, Kevin p...
    9. When Julia and Peter went to the garden, Peter hande...
   10. When Diana and Steve went to the museum, Steve showe...

----------------------------------------------------------------------
Enter a number (1-10) for example, or type your own prompt
Your choice (or 'q' to quit): 5

🔍 Analyzing: "When Lisa and David went to the beach, David threw..."
======================================================================
MECHANISTIC EXPLANATION
======================================================================

╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  DOMAIN: Indirect Object Identification (IOI) Tasks Only     ║
║  This tool analyzes sentences like:                              ║
║  "When [Name1] and [Name2] went to X, [Name2] gave Y to _"      ║
╚══════════════════════════════════════════════════════════════════╝

📝 INPUT:
   When Lisa and David went to the beach, David threw a ball to

🎯 PREDICTION:
   "Lisa" (Model Confidence: 23.6%)
   ⚠️  LOW CONFIDENCE: Model uncertainty is high; prediction reliability is reduced

👥 NAMES IDENTIFIED:
   • Indirect Object (recipient): Lisa
   • Subject (giver): David

🔬 MECHANISTIC EVIDENCE:
----------------------------------------------------------------------
   Head     | Role                   | Global Imp.  | Attention   
   -------- | ---------------------- | ------------ | ------------
   L9H9     | Name Mover (Primary)   |  17.4% (Avg) |  54.5% → Lisa
   L8H10    | S-Inhibition           |  12.3% (Avg) |   3.8% → Lisa
   L7H3     | Name Mover (Secondary) |  10.3% (Avg) |   0.8% → Lisa
   L10H6    | Backup Name Mover      |   8.9% (Avg) |  19.7% → Lisa
   L9H6     | Name Mover (Tertiary)  |   6.3% (Avg) |  61.4% → Lisa
   L10H0    | Output Head            |   6.2% (Avg) |  35.7% → Lisa

   📌 TERMINOLOGY NOTE:
   • Global Importance = Dataset-level average from our ablation experiments (following IOI methodology)
   • Attention = This specific input (correlation, NOT causation)

📊 FAITHFULNESS METRICS (ERASER-style):
----------------------------------------------------------------------
   • Sufficiency:        100.0%
     (Prediction preserved when using ONLY the 6 cited heads)
     Note: This measures performance retention, not total explanation

   • Comprehensiveness:  24.6%
     (Prediction reduced by 24.6% when ablating cited heads)
     ⚠️  LOW COVERAGE WARNING:
        75.4% of computation uses backup/distributed circuits
        The explanation is INCOMPLETE (this is a known limitation)

   • Local Faithfulness Score:           48.7%
     (ERASER-style proxy: harmonic mean of Sufficiency and Comprehensiveness)

✅ TRUST ASSESSMENT:
----------------------------------------------------------------------
   Model Confidence:     23.6%
   Sufficiency:          100.0%
   Comprehensiveness:    24.6%

   🔒 Trust Level: LOW
   Reason: Low model confidence (23.6%) - prediction may be unreliable

   ⚠️  WARNINGS:
      • LOW COVERAGE: Only 24.6% of computation explained
      • Remaining 75.4% uses distributed/backup circuits

💬 NATURAL LANGUAGE EXPLANATION:
----------------------------------------------------------------------
   The model predicts 'Lisa' because:
   
   1. The Name Mover head L9H9 attends to 'Lisa' 
      with 54.5% attention weight, copying it to the output.
   
   2. The S-Inhibition head L8H10 suppresses 'David' (the giver),
      preventing the model from outputting the subject instead.
   
   3. These 6 heads together account for 61.4% of the circuit's
      normalized logit-difference attribution captured by the predefined IOI circuit.

   ⚠️  CAVEAT: The low comprehensiveness (24.6%) indicates that
   backup circuits and distributed computation contribute significantly.
   This explanation covers the PRIMARY mechanism but not the full picture.

⚠️  LIMITATIONS:
----------------------------------------------------------------------
   • Validated only on GPT-2 Small (124M parameters)
   • Only works for Indirect Object Identification (IOI) tasks
   • Global importance scores are dataset averages, not instance-specific
   • Comprehensiveness gap indicates distributed computation exists
   • May not generalize to other models or tasks

======================================================================

----------------------------------------------------------------------
Enter a number (1-10) for example, or type your own prompt
Your choice (or 'q' to quit): 2

🔍 Analyzing: "When Alice and Bob went to the park, Bob handed a ..."
======================================================================
MECHANISTIC EXPLANATION
======================================================================

╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  DOMAIN: Indirect Object Identification (IOI) Tasks Only     ║
║  This tool analyzes sentences like:                              ║
║  "When [Name1] and [Name2] went to X, [Name2] gave Y to _"      ║
╚══════════════════════════════════════════════════════════════════╝

📝 INPUT:
   When Alice and Bob went to the park, Bob handed a flower to

🎯 PREDICTION:
   "Alice" (Model Confidence: 86.3%)

👥 NAMES IDENTIFIED:
   • Indirect Object (recipient): Alice
   • Subject (giver): Bob

🔬 MECHANISTIC EVIDENCE:
----------------------------------------------------------------------
   Head     | Role                   | Global Imp.  | Attention   
   -------- | ---------------------- | ------------ | ------------
   L9H9     | Name Mover (Primary)   |  17.4% (Avg) |  81.3% → Alice
   L8H10    | S-Inhibition           |  12.3% (Avg) |   6.9% → Alice
   L7H3     | Name Mover (Secondary) |  10.3% (Avg) |   1.0% → Alice
   L10H6    | Backup Name Mover      |   8.9% (Avg) |  23.6% → Alice
   L9H6     | Name Mover (Tertiary)  |   6.3% (Avg) |  72.1% → Alice
   L10H0    | Output Head            |   6.2% (Avg) |  43.4% → Alice

   📌 TERMINOLOGY NOTE:
   • Global Importance = Dataset-level average from our ablation experiments (following IOI methodology)
   • Attention = This specific input (correlation, NOT causation)

📊 FAITHFULNESS METRICS (ERASER-style):
----------------------------------------------------------------------
   • Sufficiency:        100.0%
     (Prediction preserved when using ONLY the 6 cited heads)
     Note: This measures performance retention, not total explanation

   • Comprehensiveness:  24.3%
     (Prediction reduced by 24.3% when ablating cited heads)
     ⚠️  LOW COVERAGE WARNING:
        75.7% of computation uses backup/distributed circuits
        The explanation is INCOMPLETE (this is a known limitation)

   • Local Faithfulness Score:           48.1%
     (ERASER-style proxy: harmonic mean of Sufficiency and Comprehensiveness)

✅ TRUST ASSESSMENT:
----------------------------------------------------------------------
   Model Confidence:     86.3%
   Sufficiency:          100.0%
   Comprehensiveness:    24.3%

   🔒 Trust Level: HIGH
   Reason: High model confidence and high sufficiency

   ⚠️  WARNINGS:
      • LOW COVERAGE: Only 24.3% of computation explained
      • Remaining 75.7% uses distributed/backup circuits

💬 NATURAL LANGUAGE EXPLANATION:
----------------------------------------------------------------------
   The model predicts 'Alice' because:
   
   1. The Name Mover head L9H9 attends to 'Alice' 
      with 81.3% attention weight, copying it to the output.
   
   2. The S-Inhibition head L8H10 suppresses 'Bob' (the giver),
      preventing the model from outputting the subject instead.
   
   3. These 6 heads together account for 61.4% of the circuit's
      normalized logit-difference attribution captured by the predefined IOI circuit.

   ⚠️  CAVEAT: The low comprehensiveness (24.3%) indicates that
   backup circuits and distributed computation contribute significantly.
   This explanation covers the PRIMARY mechanism but not the full picture.

⚠️  LIMITATIONS:
----------------------------------------------------------------------
   • Validated only on GPT-2 Small (124M parameters)
   • Only works for Indirect Object Identification (IOI) tasks
   • Global importance scores are dataset averages, not instance-specific
   • Comprehensiveness gap indicates distributed computation exists
   • May not generalize to other models or tasks

======================================================================

----------------------------------------------------------------------
Enter a number (1-10) for example, or type your own prompt
Your choice (or 'q' to quit): 7

🔍 Analyzing: "When Rachel and Chris went to the office, Chris se..."
======================================================================
MECHANISTIC EXPLANATION
======================================================================

╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  DOMAIN: Indirect Object Identification (IOI) Tasks Only     ║
║  This tool analyzes sentences like:                              ║
║  "When [Name1] and [Name2] went to X, [Name2] gave Y to _"      ║
╚══════════════════════════════════════════════════════════════════╝

📝 INPUT:
   When Rachel and Chris went to the office, Chris sent a message to

🎯 PREDICTION:
   "Rachel" (Model Confidence: 54.6%)

👥 NAMES IDENTIFIED:
   • Indirect Object (recipient): Rachel
   • Subject (giver): Chris

🔬 MECHANISTIC EVIDENCE:
----------------------------------------------------------------------
   Head     | Role                   | Global Imp.  | Attention   
   -------- | ---------------------- | ------------ | ------------
   L9H9     | Name Mover (Primary)   |  17.4% (Avg) |  79.2% → Rachel
   L8H10    | S-Inhibition           |  12.3% (Avg) |   4.6% → Rachel
   L7H3     | Name Mover (Secondary) |  10.3% (Avg) |   2.1% → Rachel
   L10H6    | Backup Name Mover      |   8.9% (Avg) |  14.8% → Rachel
   L9H6     | Name Mover (Tertiary)  |   6.3% (Avg) |  75.2% → Rachel
   L10H0    | Output Head            |   6.2% (Avg) |  35.6% → Rachel

   📌 TERMINOLOGY NOTE:
   • Global Importance = Dataset-level average from our ablation experiments (following IOI methodology)
   • Attention = This specific input (correlation, NOT causation)

📊 FAITHFULNESS METRICS (ERASER-style):
----------------------------------------------------------------------
   • Sufficiency:        100.0%
     (Prediction preserved when using ONLY the 6 cited heads)
     Note: This measures performance retention, not total explanation

   • Comprehensiveness:  31.0%
     (Prediction reduced by 31.0% when ablating cited heads)

   • Local Faithfulness Score:           61.5%
     (ERASER-style proxy: harmonic mean of Sufficiency and Comprehensiveness)

✅ TRUST ASSESSMENT:
----------------------------------------------------------------------
   Model Confidence:     54.6%
   Sufficiency:          100.0%
   Comprehensiveness:    31.0%

   🔒 Trust Level: MEDIUM
   Reason: Moderate confidence and sufficiency

💬 NATURAL LANGUAGE EXPLANATION:
----------------------------------------------------------------------
   The model predicts 'Rachel' because:
   
   1. The Name Mover head L9H9 attends to 'Rachel' 
      with 79.2% attention weight, copying it to the output.
   
   2. The S-Inhibition head L8H10 suppresses 'Chris' (the giver),
      preventing the model from outputting the subject instead.
   
   3. These 6 heads together account for 61.4% of the circuit's
      normalized logit-difference attribution captured by the predefined IOI circuit.

⚠️  LIMITATIONS:
----------------------------------------------------------------------
   • Validated only on GPT-2 Small (124M parameters)
   • Only works for Indirect Object Identification (IOI) tasks
   • Global importance scores are dataset averages, not instance-specific
   • Comprehensiveness gap indicates distributed computation exists
   • May not generalize to other models or tasks

======================================================================

----------------------------------------------------------------------
Enter a number (1-10) for example, or type your own prompt
Your choice (or 'q' to quit): 1

🔍 Analyzing: "When Mary and John went to the store, John gave a ..."
======================================================================
MECHANISTIC EXPLANATION
======================================================================

╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  DOMAIN: Indirect Object Identification (IOI) Tasks Only     ║
║  This tool analyzes sentences like:                              ║
║  "When [Name1] and [Name2] went to X, [Name2] gave Y to _"      ║
╚══════════════════════════════════════════════════════════════════╝

📝 INPUT:
   When Mary and John went to the store, John gave a drink to

🎯 PREDICTION:
   "Mary" (Model Confidence: 67.7%)

👥 NAMES IDENTIFIED:
   • Indirect Object (recipient): Mary
   • Subject (giver): John

🔬 MECHANISTIC EVIDENCE:
----------------------------------------------------------------------
   Head     | Role                   | Global Imp.  | Attention   
   -------- | ---------------------- | ------------ | ------------
   L9H9     | Name Mover (Primary)   |  17.4% (Avg) |  66.5% → Mary
   L8H10    | S-Inhibition           |  12.3% (Avg) |   5.8% → Mary
   L7H3     | Name Mover (Secondary) |  10.3% (Avg) |   0.6% → Mary
   L10H6    | Backup Name Mover      |   8.9% (Avg) |  10.8% → Mary
   L9H6     | Name Mover (Tertiary)  |   6.3% (Avg) |  67.2% → Mary
   L10H0    | Output Head            |   6.2% (Avg) |  33.2% → Mary

   📌 TERMINOLOGY NOTE:
   • Global Importance = Dataset-level average from our ablation experiments (following IOI methodology)
   • Attention = This specific input (correlation, NOT causation)

📊 FAITHFULNESS METRICS (ERASER-style):
----------------------------------------------------------------------
   • Sufficiency:        100.0%
     (Prediction preserved when using ONLY the 6 cited heads)
     Note: This measures performance retention, not total explanation

   • Comprehensiveness:  28.4%
     (Prediction reduced by 28.4% when ablating cited heads)
     ⚠️  LOW COVERAGE WARNING:
        71.6% of computation uses backup/distributed circuits
        The explanation is INCOMPLETE (this is a known limitation)

   • Local Faithfulness Score:           56.2%
     (ERASER-style proxy: harmonic mean of Sufficiency and Comprehensiveness)

✅ TRUST ASSESSMENT:
----------------------------------------------------------------------
   Model Confidence:     67.7%
   Sufficiency:          100.0%
   Comprehensiveness:    28.4%

   🔒 Trust Level: MEDIUM
   Reason: Adequate confidence but low comprehensiveness suggests backup circuits

   ⚠️  WARNINGS:
      • LOW COVERAGE: Only 28.4% of computation explained
      • Remaining 71.6% uses distributed/backup circuits

💬 NATURAL LANGUAGE EXPLANATION:
----------------------------------------------------------------------
   The model predicts 'Mary' because:
   
   1. The Name Mover head L9H9 attends to 'Mary' 
      with 66.5% attention weight, copying it to the output.
   
   2. The S-Inhibition head L8H10 suppresses 'John' (the giver),
      preventing the model from outputting the subject instead.
   
   3. These 6 heads together account for 61.4% of the circuit's
      normalized logit-difference attribution captured by the predefined IOI circuit.

   ⚠️  CAVEAT: The low comprehensiveness (28.4%) indicates that
   backup circuits and distributed computation contribute significantly.
   This explanation covers the PRIMARY mechanism but not the full picture.

⚠️  LIMITATIONS:
----------------------------------------------------------------------
   • Validated only on GPT-2 Small (124M parameters)
   • Only works for Indirect Object Identification (IOI) tasks
   • Global importance scores are dataset averages, not instance-specific
   • Comprehensiveness gap indicates distributed computation exists
   • May not generalize to other models or tasks

======================================================================

----------------------------------------------------------------------
Enter a number (1-10) for example, or type your own prompt
Your choice (or 'q' to quit): q
Goodbye!

> ⚠️ Note: Low comprehensiveness (~22%) indicates backup circuits contribute significantly. This is a known limitation.
````



---

## 📊 Metric Definitions

| Metric                        | Definition                                                              |
|----------------------------|----------------------------------------------------------------------------|
| Sufficiency                | Prediction preserved when using ONLY the cited heads                       |
| Comprehensiveness          | Prediction reduction when ABLATING the cited heads                         |
| Local Faithfulness Score   | ERASER-style proxy: harmonic mean of Sufficiency and Comprehensiveness     |
| Global Importance          | Dataset-level average from ablation experiments (following IOI methodology) |
| Attention Weight           | Instance-level correlation (NOT causation)                                 |

---

## ⚠️ Limitations

- Model: Validated only on GPT-2 Small (124M parameters)
- Task: Only works for Indirect Object Identification (IOI) tasks
- Generalization: Results may not generalize to larger models or other tasks
- Comprehensiveness: 22% average indicates significant distributed computation

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
