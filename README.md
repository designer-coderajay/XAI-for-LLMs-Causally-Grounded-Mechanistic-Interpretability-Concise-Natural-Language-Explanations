# Causally Grounded Mechanistic Interpretability and Faithful Natural-Language Explanations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## MSc Thesis Project

**Title:** Causally Grounded Mechanistic Interpretability for LLMs with Faithful Natural-Language Explanations
**Author:** Ajay Pravin Mahale (Matrikel 979783)
**Institution:** Hochschule Trier, Fachbereich Technik
**Supervisor:** Prof. Dr. Ernst Georg Haffner
**Contact:** jymh0144@hochschule-trier.de
**Submission:** May 2026

---

## Overview

This repository contains the experimental code for an MSc thesis on generating causally faithful natural-language explanations from mechanistic circuit analysis of transformer language models. The work is restricted to the Indirect Object Identification (IOI) task on GPT-2 Small and should not be interpreted as a general LLM explanation framework.

### Key contributions

1. **Circuit-to-NL pipeline.** Translates mechanistic circuit findings into natural-language explanations, evaluated end-to-end with ERASER faithfulness metrics.
2. **ERASER adaptation.** First systematic application of ERASER sufficiency and comprehensiveness metrics to mechanistic interpretability rather than rationale extraction.
3. **Template vs LLM comparison.** Quantifies what fluency adds (and what it does not). Same circuit data, two generators, +66% relative quality improvement, zero faithfulness improvement.
4. **Failure taxonomy.** Three-way classification (distributed computation, missing cited head, redundant head activity) of when and why explanations diverge from mechanisms.

### Results summary

All numbers below are from the thesis (Table 1 and Section 4), `n = 50` for the main evaluation, `n = 30` for the template-vs-LLM comparison.

| Metric | Value |
|---|---|
| IOI circuit coverage (6 heads) | 61.4% of logit difference |
| Sufficiency (mean) | 100.0% |
| Comprehensiveness (mean) | 22.0% |
| F1 (harmonic mean) | 36.0% |
| Beats attention baseline | in 75% of prompts |
| LLM vs template quality | +66% relative improvement (60% to 99%) |
| Confidence × comprehensiveness | r = 0.009 |
| L10H10 appears in failures | 82% |

### Three findings worth highlighting

- **Sufficient does not mean necessary.** The six-head circuit retains the full prediction on its own (100% sufficiency) but removing it only drops the prediction by 22%. Backup mechanisms cover the rest. Reporting only sufficiency overstates what the explanation establishes.
- **Format is not faithfulness.** Same circuit data, template version scores 60% on quality, LLM version scores 99%. A beautifully written explanation can sit on top of a 22%-comprehensive circuit. Fluency and grounding are independent axes.
- **Confidence is not a shortcut.** Correlation between model confidence and explanation faithfulness is r = 0.009. Confidence tells you nothing about whether the explanation reflects the mechanism.

---

## Repository structure
