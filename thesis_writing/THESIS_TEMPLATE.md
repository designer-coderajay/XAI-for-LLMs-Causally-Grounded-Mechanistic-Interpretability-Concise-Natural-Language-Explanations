# MASTER'S THESIS

## Explainable AI for LLMs: Causally Grounded Mechanistic Interpretability and Concise Natural-Language Explanations

**Author:** Ajay  
**Supervisor:** Prof. Dr. Haffner  
**Institution:** Hochschule Trier  
**Date:** May 2026

---

# STRUCTURE OVERVIEW (12-15 Pages)

| Section | Pages | Status |
|---------|-------|--------|
| Abstract | 0.5 | ⬜ |
| 1. Introduction | 1.5 | ⬜ |
| 2. Related Work | 2.0 | ⬜ |
| 3. Methodology | 3.5 | ⬜ |
| 4. Results | 3.5 | ⬜ |
| 5. Discussion | 2.0 | ⬜ |
| 6. Conclusion | 0.5 | ⬜ |
| References | 1.0 | ⬜ |
| **TOTAL** | **14.5** | |

---

# ABSTRACT (0.5 page)

## What To Write:
- Problem (1-2 sentences)
- Gap (1 sentence)
- Your method (2-3 sentences)
- Results (2-3 sentences)
- Conclusion (1 sentence)

## Template:

Large Language Models (LLMs) achieve remarkable performance but remain opaque in their decision-making. Existing explanation methods either provide unfaithful rationales or produce technical outputs inaccessible to non-experts. This thesis presents a novel approach that translates mechanistic circuit analysis into natural language explanations, evaluated using causal faithfulness metrics.

We analyze GPT-2 Small on the Indirect Object Identification (IOI) task, identifying key attention heads responsible for the model's predictions. We then generate natural language explanations using both template-based and LLM-based approaches, evaluating faithfulness using ERASER metrics (sufficiency and comprehensiveness).

Our circuit-based method achieves 100% sufficiency and 36% F1 score, outperforming attention-based baselines by 75%. LLM-generated explanations score 98% quality compared to 60% for templates, representing a 63% improvement. We find that 34% of cases exhibit distributed computation beyond the primary circuit, providing insights into when explanations diverge from internal mechanisms.

---

# 1. INTRODUCTION (1.5 pages)

## Structure:
1. **Context** (1 paragraph): LLMs are powerful but opaque
2. **Problem** (1 paragraph): Current explanations are unfaithful or inaccessible
3. **Gap** (1 paragraph): No method combines mechanistic interp + NL + causal evaluation
4. **Contribution** (1 paragraph): What you did
5. **Research Questions** (list)
6. **Thesis Structure** (1 paragraph)

## What To Write:

### 1.1 Context
Large Language Models have achieved remarkable success across natural language tasks, from translation to code generation. However, their decision-making processes remain largely opaque, creating challenges for trust, debugging, and safety-critical applications. Understanding why a model makes specific predictions is crucial for deploying these systems responsibly.

### 1.2 Problem
Existing approaches to explaining LLM behavior fall into two categories. Post-hoc methods like SHAP and LIME provide input attributions but lack causal grounding—they show correlations, not mechanisms. Mechanistic interpretability methods identify internal circuits but produce technical outputs (attention patterns, activation values) inaccessible to non-experts. Neither approach provides faithful, human-readable explanations.

### 1.3 Gap
To our knowledge, no prior work systematically combines: (1) mechanistic circuit identification, (2) natural language explanation generation, and (3) causal faithfulness evaluation. This gap prevents practitioners from obtaining explanations that are both accurate and understandable.

### 1.4 Contribution
This thesis addresses this gap with three contributions:
1. We reproduce and validate the IOI circuit in GPT-2 Small using activation patching
2. We develop a pipeline that generates NL explanations from circuit analysis
3. We evaluate faithfulness using ERASER metrics and compare template vs LLM-generated explanations

### 1.5 Research Questions
- **RQ1:** Which model components are causally responsible for IOI task performance?
- **RQ2:** Can mechanistic signals be translated into causally faithful NL explanations?
- **RQ3:** When and why do NL explanations diverge from internal mechanisms?

### 1.6 Thesis Structure
Section 2 reviews related work. Section 3 describes our methodology. Section 4 presents results. Section 5 discusses findings and limitations. Section 6 concludes.

---

# 2. RELATED WORK (2 pages)

## Structure:
1. **Mechanistic Interpretability** (0.5 page)
2. **Natural Language Explanations** (0.5 page)
3. **Faithfulness Evaluation** (0.5 page)
4. **Gap & Your Position** (0.5 page)

## What To Write:

### 2.1 Mechanistic Interpretability
Mechanistic interpretability aims to reverse-engineer neural network computations into human-understandable algorithms. Elhage et al. (2021) introduced the concept of circuits—subgraphs of model components that implement specific behaviors. Wang et al. (2023) identified the Indirect Object Identification (IOI) circuit in GPT-2, showing how Name Mover and S-Inhibition heads collaborate to predict indirect objects. Conmy et al. (2023) developed automated circuit discovery methods. This work provides the foundation for our circuit analysis.

### 2.2 Natural Language Explanations for Neural Networks
Several approaches generate NL explanations for model behavior. Bills et al. (2023) used GPT-4 to automatically describe neuron activations, finding that language models can explain individual neurons. Singh et al. (2024) developed SASC, which generates explanations via optimization. Camburu et al. (2018) introduced e-SNLI, providing human explanations for natural language inference. We build on these approaches by generating explanations specifically from mechanistic circuit data.

### 2.3 Faithfulness Evaluation
DeYoung et al. (2020) introduced ERASER, a benchmark for evaluating rationale faithfulness using sufficiency and comprehensiveness metrics. Sufficiency measures whether cited evidence alone predicts the output. Comprehensiveness measures whether removing cited evidence changes the prediction. Jacovi & Goldberg (2020) argued that faithfulness—not plausibility—should be the primary evaluation criterion. We adopt ERASER metrics for evaluating our NL explanations.

### 2.4 Gap and Our Contribution
| Prior Work | Limitation |
|------------|------------|
| Wang et al. (IOI) | No NL explanations |
| Bills et al. | Neurons, not circuits |
| e-SNLI | Human explanations, not automated |
| ERASER | Input rationales, not circuit explanations |

Our work bridges these: Circuit analysis → NL generation → Faithfulness evaluation.

## Citations Needed (5-8):
1. Wang et al. (2023) - IOI circuit
2. Bills et al. (2023) - Neuron explanations
3. DeYoung et al. (2020) - ERASER metrics
4. Camburu et al. (2018) - e-SNLI
5. Elhage et al. (2021) - Circuits
6. Jacovi & Goldberg (2020) - Faithfulness
7. Singh et al. (2024) - SASC (optional)
8. Conmy et al. (2023) - ACDC (optional)

---

# 3. METHODOLOGY (3.5 pages)

## Structure:
1. **Task & Model** (0.5 page)
2. **Circuit Identification** (1 page)
3. **NL Explanation Generation** (1 page)
4. **Faithfulness Evaluation** (1 page)

## What To Write:

### 3.1 Task and Model

**Model:** We use GPT-2 Small (124M parameters, 12 layers, 12 attention heads per layer) implemented via TransformerLens (Nanda, 2022).

**Task:** The Indirect Object Identification (IOI) task tests whether models correctly identify indirect objects in sentences like:

> "When Mary and John went to the store, John gave a drink to ___"

The correct completion is "Mary" (indirect object), not "John" (subject). This task isolates a specific linguistic capability with known circuit-level implementation.

**Dataset:** We generate 50 IOI prompts using 25 name pairs and 2 sentence templates:
- Template 1: "When {name1} and {name2} went to the store, {name2} gave a drink to"
- Template 2: "When {name1} and {name2} went to the park, {name2} handed a flower to"

### 3.2 Circuit Identification

We identify the IOI circuit using activation patching (causal intervention):

1. **Clean run:** Process IOI prompt, cache all activations
2. **Corrupted run:** Swap name positions, cache activations
3. **Patching:** Restore each head's activation from clean to corrupted run
4. **Measurement:** Calculate recovery of logit difference (IO token - Subject token)

**Logit Difference:** 
$$\text{LogitDiff} = \text{logit}(\text{IO}) - \text{logit}(\text{Subject})$$

**Patching Effect:**
$$\text{Effect}_h = \frac{\text{LogitDiff}_{\text{patched}} - \text{LogitDiff}_{\text{corrupted}}}{\text{LogitDiff}_{\text{clean}} - \text{LogitDiff}_{\text{corrupted}}}$$

[INSERT FIGURE 1: Circuit Heatmap - ioi_circuit_heatmap.png]

### 3.3 Natural Language Explanation Generation

We implement two explanation methods:

**Template-Based (Baseline):**
Fixed template filled with extracted values:
> "The model predicts '{prediction}' because L9H9 and L9H6 attend to it with high attention, copying the indirect object to output position."

**LLM-Generated (Novel):**
We provide circuit data to Claude and prompt it to generate mechanistically-grounded explanations:
```
INPUT: "{prompt}"
PREDICTION: "{prediction}" (confidence: {confidence})
MECHANISTIC DATA:
- L9H9 attention: {attention_L9H9}
- L9H6 attention: {attention_L9H6}
Generate a 1-2 sentence explanation mentioning specific heads and percentages.
```

### 3.4 Faithfulness Evaluation (ERASER Metrics)

**Sufficiency:** Do cited heads alone explain the prediction?
$$\text{Sufficiency} = \frac{\sum_{h \in \text{cited}} \text{Contribution}_h}{\text{LogitDiff}_{\text{clean}}}$$

**Comprehensiveness:** Does removing cited heads change the prediction?
$$\text{Comprehensiveness} = 1 - \frac{\text{LogitDiff}_{\text{ablated}}}{\text{LogitDiff}_{\text{clean}}}$$

**F1 Score:**
$$F1 = \frac{2 \times \text{Sufficiency} \times \text{Comprehensiveness}}{\text{Sufficiency} + \text{Comprehensiveness}}$$

---

# 4. RESULTS (3.5 pages)

## Structure:
1. **RQ1: Circuit Identification** (1 page)
2. **RQ2: Faithfulness Evaluation** (1.5 pages)
3. **RQ3: Failure Analysis** (1 page)

## What To Write:

### 4.1 RQ1: Circuit Identification

GPT-2 Small achieves 100% accuracy on the IOI task (50/50 prompts). Activation patching identifies six key attention heads:

[INSERT TABLE 1: Circuit Heads]

| Head | Role | Causal Effect |
|------|------|---------------|
| L9H9 | Name Mover | 17.4% |
| L8H10 | S-Inhibition | 12.3% |
| L7H3 | S-Inhibition | 10.3% |
| L10H6 | Backup Name Mover | 8.9% |
| L9H6 | Name Mover | 6.3% |
| L10H0 | Backup Name Mover | 6.2% |

These heads account for 61.4% of the total logit difference, consistent with Wang et al. (2023). L9H9 (Name Mover) shows highest causal importance, attending to the indirect object with 66.5% average attention.

### 4.2 RQ2: Faithfulness Evaluation

[INSERT TABLE 2: ERASER Metrics]

| Metric | Value |
|--------|-------|
| Sufficiency | 100.0% ± 0.0% |
| Comprehensiveness | 22.0% ± 17.3% |
| F1 Score | 36.0% |

Our circuit-based explanations achieve perfect sufficiency, indicating the cited heads fully account for the prediction. Comprehensiveness of 22% suggests additional distributed computation beyond the primary circuit.

[INSERT TABLE 3: Baseline Comparison]

| Method | Sufficiency | Comprehensiveness | F1 |
|--------|-------------|-------------------|-----|
| Ours (Circuit) | 100.0% | 22.0% | 36.0% |
| Attention-Based | 16.7% | 26.6% | 20.6% |
| Random | 50.7% | 24.3% | 32.8% |

Our method outperforms attention-based baseline by 75% on F1 score.

[INSERT FIGURE 2: Baseline Comparison Chart - baseline_comparison.png]

**Template vs LLM-Generated Explanations:**

[INSERT TABLE 4: Explanation Quality]

| Metric | Template | Learned (Claude) | Improvement |
|--------|----------|------------------|-------------|
| Quality Score | 60% | 98% | +63% |
| Uses Actual Percentages | 0% | 100% | +100% |
| Mentions Both Names | 0% | 90% | +90% |
| Avg Word Count | 21 | 52 | 2.5x |

LLM-generated explanations are significantly more specific and informative.

**Example Comparison:**

*Template:* "The model predicts 'Mary' because L9H9 and L9H6 attend to it with high attention, copying the indirect object to output position."

*LLM-Generated:* "GPT-2 predicts 'Mary' by identifying her as the indirect object through L9H9 (66.5% attention to Mary) and L9H6 (67.2% attention to Mary), which strongly focus on Mary while giving minimal attention to John (7.0% and 10.6% respectively)."

### 4.3 RQ3: Failure Analysis

[INSERT TABLE 5: Failure Categories]

| Category | Count | Percentage |
|----------|-------|------------|
| High comprehensiveness (>25%) | 18 | 36% |
| Near threshold (15-25%) | 15 | 30% |
| Low comprehensiveness (<15%) | 17 | 34% |

34% of cases show low comprehensiveness, indicating the circuit explanation doesn't fully capture the mechanism. Analysis reveals:

1. **Distributed computation:** L10H10 appears in 82% of cases but isn't in our primary circuit
2. **No confidence correlation:** r = 0.009 between model confidence and comprehensiveness
3. **Optimal circuit size:** Adding L10H10 doesn't improve F1 (34.4% vs 36.0%)

---

# 5. DISCUSSION (2 pages)

## Structure:
1. **Summary of Findings** (0.5 page)
2. **Implications** (0.5 page)
3. **Limitations** (0.5 page)
4. **Future Work** (0.5 page)

## What To Write:

### 5.1 Summary of Findings

This thesis demonstrates that mechanistic circuit analysis can be translated into causally faithful natural language explanations. Three key findings emerge:

1. **Circuit identification works:** Activation patching successfully identifies the IOI circuit, with L9H9 as the primary Name Mover head (17.4% causal effect).

2. **LLM-generated explanations outperform templates:** Claude-generated explanations achieve 98% quality vs 60% for templates, by including specific attention percentages and contextual reasoning.

3. **Explanations have limits:** 34% of cases show distributed computation beyond the primary circuit, indicating that explanations should acknowledge uncertainty.

### 5.2 Implications

**For Practitioners:** Our pipeline provides a method to generate human-readable explanations for model predictions, grounded in causal analysis rather than correlation.

**For Researchers:** The gap between sufficiency (100%) and comprehensiveness (22%) suggests that neural network computation is more distributed than modular circuit analysis implies.

**For AI Safety:** Faithful explanations are crucial for trust. Our ERASER-based evaluation provides a principled way to measure faithfulness.

### 5.3 Limitations

1. **Single task:** We evaluate only on IOI. Generalization to other tasks is untested.

2. **Single model:** GPT-2 Small (124M parameters). Larger models may have different circuit structures.

3. **Prompt scale:** 50 prompts provide statistical power but larger evaluation would strengthen confidence.

4. **No SHAP/LIME comparison:** Technical constraints prevented direct comparison with these common baselines.

5. **LLM dependency:** Our novel explanation method requires API access to Claude, adding cost and latency.

### 5.4 Future Work

1. **Multi-task evaluation:** Apply our pipeline to Greater-Than, Docstring, and other tasks with known circuits.

2. **Larger models:** Investigate whether circuits and explanations transfer to GPT-2 Medium/Large or GPT-Neo.

3. **Learned explanation generation:** Train a smaller model to generate explanations, removing API dependency.

4. **User studies:** Evaluate whether humans find LLM-generated explanations more useful than templates.

---

# 6. CONCLUSION (0.5 page)

## What To Write:

This thesis presented a novel approach to generating faithful natural language explanations for LLM predictions by combining mechanistic interpretability with LLM-based explanation generation.

We identified the IOI circuit in GPT-2 Small through activation patching, achieving 100% sufficiency on ERASER metrics. Our circuit-based method outperforms attention-based baselines by 75%. LLM-generated explanations score 98% quality compared to 60% for templates, demonstrating that mechanistic data enables more specific and informative explanations.

We also found that 34% of cases exhibit distributed computation beyond the primary circuit, providing guidance on when to express uncertainty in explanations.

This work contributes to the broader goal of making AI systems more transparent and trustworthy by bridging the gap between technical interpretability research and human-understandable explanations.

---

# REFERENCES

## Format (IEEE or APA, check with Prof):

[1] Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the wild: A circuit for indirect object identification in GPT-2 Small. ICLR.

[2] Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., ... & Olah, C. (2023). Language models can explain neurons in language models. OpenAI Blog.

[3] DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). ERASER: A benchmark to evaluate rationalized NLP models. ACL.

[4] Camburu, O. M., Rocktäschel, T., Lukasiewicz, T., & Blunsom, P. (2018). e-SNLI: Natural language inference with natural language explanations. NeurIPS.

[5] Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C. (2021). A mathematical framework for transformer circuits. Anthropic.

[6] Jacovi, A., & Goldberg, Y. (2020). Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? ACL.

[7] Nanda, N. (2022). TransformerLens. GitHub repository.

[8] Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards automated circuit discovery for mechanistic interpretability. NeurIPS.

---

# FIGURES TO INCLUDE

## Figure 1: IOI Circuit Heatmap
- File: results/figures/ioi_circuit_heatmap.png
- Caption: "Causal importance of attention heads for IOI task. L9H9 shows highest effect (17.4%)."
- Place in: Section 3.2 or 4.1

## Figure 2: Baseline Comparison
- File: results/figures/baseline_comparison.png
- Caption: "ERASER faithfulness metrics across methods. Our circuit-based approach outperforms baselines."
- Place in: Section 4.2

---

# TABLES TO INCLUDE

## Table 1: Circuit Heads
- Place in: Section 4.1

## Table 2: ERASER Metrics
- Place in: Section 4.2

## Table 3: Baseline Comparison
- Place in: Section 4.2

## Table 4: Template vs Learned
- Place in: Section 4.2

## Table 5: Failure Categories
- Place in: Section 4.3

---

# WRITING TIPS FOR GRADE 1.0

## Do:
- Use precise language ("achieves 100% sufficiency" not "does well")
- Include exact numbers from your experiments
- Cite sources for every claim about prior work
- Be honest about limitations
- Use consistent terminology throughout

## Don't:
- Use vague language ("seems to work", "kind of")
- Make claims without evidence
- Hide limitations
- Use first person excessively (prefer "we" over "I")
- Include unnecessary background

## Style:
- Short paragraphs (3-5 sentences)
- Active voice when possible
- One idea per paragraph
- Clear topic sentences

## Checklist Before Submission:
- [ ] All RQs answered explicitly
- [ ] All figures have captions
- [ ] All tables have titles
- [ ] All citations in reference list
- [ ] Page count within limit
- [ ] Proofread for grammar
- [ ] Consistent formatting

