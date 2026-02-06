# 🔍 RELATED WORK ANALYSIS

## MSc Thesis: Explainable AI for LLMs via Mechanistic Interpretability
**Author:** Ajay Mahale

---

## ⚠️ CRITICAL: Bills et al. 2023 (OpenAI)

### Paper: "Language Models Can Explain Neurons in Language Models"

This is your **CLOSEST related work**. You MUST address this in your thesis.

---

## 📊 DETAILED COMPARISON

| Aspect | Bills et al. (2023) | Your Work |
|--------|---------------------|-----------|
| **Granularity** | Individual neurons | Circuits (multiple heads) |
| **Model Explained** | GPT-2 | GPT-2 |
| **Explainer Model** | GPT-4 | Claude (Sonnet) |
| **Explanation Target** | What a neuron detects | Why a prediction is made |
| **Evaluation Method** | Simulation scoring | ERASER (sufficiency, comprehensiveness) |
| **Ground Truth** | Activation patterns | Causal interventions |
| **Task** | General (all neurons) | Specific (IOI task) |
| **Output** | "Neuron X detects Y" | "Model predicts X because heads A,B,C..." |

---

## 🎯 HOW YOUR WORK DIFFERS

### 1. DIFFERENT GRANULARITY

**Bills et al.:**
```
"Neuron 47 in layer 5 activates on words related to sports"
```

**Your work:**
```
"The model predicts 'Mary' because the Name Mover circuit (L9H9, L8H10, L7H3) 
attends to and copies the indirect object, while S-Inhibition heads suppress 
the subject."
```

**Why this matters:** Circuits provide task-level explanations (why THIS prediction), 
not feature-level explanations (what this neuron detects generally).

---

### 2. DIFFERENT EVALUATION

**Bills et al. (Simulation Scoring):**
1. Generate explanation for neuron
2. Ask GPT-4 to predict neuron activations based on explanation
3. Score: correlation between predicted and actual activations

**Your work (ERASER Metrics):**
1. Generate explanation citing specific heads
2. Sufficiency: Do cited heads explain the prediction?
3. Comprehensiveness: Are cited heads necessary?
4. Use causal interventions (ablation), not correlation

**Why this matters:** Causal evaluation > correlational evaluation

---

### 3. DIFFERENT GROUND TRUTH

**Bills et al.:**
- No independent ground truth
- Explanation quality = how well it predicts activations
- Circular: LLM explains → LLM evaluates

**Your work:**
- Ground truth: Wang et al. (2022) IOI circuit
- Known heads, known roles, known causal importance
- Independent validation possible

**Why this matters:** You can verify explanations against established circuit knowledge

---

### 4. DIFFERENT SCOPE

**Bills et al.:**
- Explains ALL neurons (exhaustive but shallow)
- No task focus
- ~300,000 neuron explanations

**Your work:**
- Explains ONE well-understood task (deep)
- IOI task with known circuit
- ~50 prompt evaluations with full causal analysis

**Why this matters:** Depth vs breadth tradeoff - you sacrifice coverage for rigor

---

## ✅ YOUR GENUINE CONTRIBUTIONS (vs Bills et al.)

| Contribution | Novel? | Why |
|--------------|--------|-----|
| Circuit-level explanations | ✅ YES | They only do neurons |
| ERASER adaptation to mech interp | ✅ YES | They use simulation scoring |
| Template vs Learned comparison | ✅ YES | They don't compare methods |
| Failure analysis (when/why diverge) | ✅ YES | They don't analyze failures |
| IOI-grounded evaluation | ✅ YES | They don't use known circuits |

---

## 📝 HOW TO WRITE THIS IN YOUR THESIS

### Chapter 2: Related Work

```
2.3 LLM-Generated Model Explanations

The most closely related work to ours is Bills et al. (2023), who demonstrated 
that large language models can generate natural language explanations for 
individual neurons in transformer models. Using GPT-4 as an "explainer," they 
generated descriptions for all neurons in GPT-2 Small, evaluated via simulation 
scoring—measuring how well the explanations predict neuron activations.

Our work differs fundamentally in four aspects:

First, we explain at the CIRCUIT level rather than neuron level. While Bills et al. 
describe what individual neurons detect ("sports-related words"), we explain 
why specific predictions are made ("the model outputs 'Mary' because the Name 
Mover heads copy the indirect object"). This provides task-relevant explanations 
rather than feature catalogs.

Second, we employ ERASER-style faithfulness metrics (DeYoung et al., 2020) 
instead of simulation scoring. Our evaluation uses causal interventions 
(activation patching, ablation) to measure whether cited components are 
sufficient and necessary for predictions. This grounds our evaluation in 
causality rather than correlation.

Third, we anchor our work to the well-characterized IOI circuit (Wang et al., 2022), 
providing independent ground truth for validation. Bills et al. evaluate 
explanations solely by how well they predict activations, creating a potentially 
circular evaluation where LLMs both generate and validate explanations.

Fourth, we systematically compare template-based and learned (LLM-generated) 
explanations, demonstrating that learned explanations achieve significantly 
higher quality (+63%) while maintaining causal faithfulness. We also analyze 
failure cases where explanations diverge from mechanistic ground truth, 
identifying distributed computation as a key challenge.

These contributions establish a framework for causally-grounded natural language 
explanations of model behavior, complementing neuron-level approaches with 
circuit-level understanding.
```

---

## ⚠️ WHAT YOU CANNOT CLAIM

| Claim | Can Say? | Why |
|-------|----------|-----|
| "First to use LLMs for model explanation" | ❌ NO | Bills et al. did this |
| "First mechanistic interpretability work" | ❌ NO | Wang et al., Elhage et al. |
| "First to generate NL explanations" | ❌ NO | Many prior works |
| "Superior to Bills et al." | ❌ NO | Different scope, not comparable |

---

## ✅ WHAT YOU CAN CLAIM

| Claim | Can Say? | Justification |
|-------|----------|---------------|
| "First ERASER evaluation of mech interp" | ✅ YES | Novel adaptation |
| "First circuit-level LLM explanations" | ✅ YES | Bills = neurons, you = circuits |
| "First template vs learned comparison" | ✅ YES | Quantitative framework |
| "To our knowledge, first systematic..." | ✅ YES | Appropriately humble |

---

## 📊 POSITIONING DIAGRAM

```
                    FEATURE LEVEL          TASK LEVEL
                    (what it detects)      (why this prediction)
                    ───────────────────────────────────────────
CORRELATION-BASED   │ Bills et al.        │ Attention-based   │
(simulation scoring)│ (2023)              │ methods           │
                    │ ✓ Neurons           │                   │
                    ├─────────────────────┼───────────────────┤
CAUSAL              │ Activation          │ YOUR WORK         │
(intervention-based)│ maximization        │ ✓ Circuits        │
                    │                     │ ✓ ERASER          │
                    │                     │ ✓ NL Explanations │
                    ───────────────────────────────────────────
```

Your work occupies the **CAUSAL + TASK LEVEL** quadrant, which is distinct from Bills et al.

---

## 🎓 THESIS DEFENSE PREPARATION

### If examiner asks: "How is this different from Bills et al.?"

**Your answer:**
> "Bills et al. explain individual neurons - what features they detect. I explain 
> circuits - why specific predictions are made. They use simulation scoring, I use 
> ERASER causal metrics. They have no independent ground truth, I validate against 
> the known IOI circuit from Wang et al. These are complementary approaches at 
> different levels of analysis."

### If examiner asks: "Why not just use their approach?"

**Your answer:**
> "Neuron-level explanations tell you what a model CAN represent, not what it 
> actually USES for a specific prediction. For explainability, we need to know 
> why THIS output was produced, which requires circuit-level analysis with 
> causal evaluation. That's what my work provides."
