# 📖 Thesis Study Guide

## Learning Roadmap: Zero to Expert

This guide helps you understand your thesis by studying it backwards (reverse engineering).

---

## Phase 1: Understand What You Built (Week 1)

### Day 1-2: Results First
1. Open `10_final_evaluation.ipynb`
2. Read all the results tables
3. Answer: What did I prove?

### Day 3-4: Novel Contribution
1. Open `08_learned_nl_generator.ipynb` and `09_template_vs_learned.ipynb`
2. Understand: Why is learned better than template?
3. Answer: What makes this novel?

### Day 5-7: Core Experiments
1. Open `02_ioi_reproduction.ipynb`
2. Trace: How did I find the circuit?
3. Open `06_failure_analysis.ipynb`
4. Understand: When do explanations fail?

---

## Phase 2: Understand the Theory (Week 2-3)

### Watch These Videos (In Order)
1. **3Blue1Brown - Neural Networks** (4 hours)
   - Understand backpropagation
   - Understand gradient descent

2. **Andrej Karpathy - GPT from Scratch** (2 hours)
   - Understand transformers
   - Understand attention

3. **Neel Nanda - Mechanistic Interpretability** (6 hours)
   - Understand circuits
   - Understand activation patching
   - Understand IOI task

### Read These Papers
1. Wang et al. - IOI Circuit (your RQ1)
2. ERASER paper - Faithfulness metrics (your evaluation)
3. Bills et al. - Neuron explanations (related work)

---

## Phase 3: Deep Dive into Code (Week 4)

### For Each Notebook, Answer:
1. What is the INPUT?
2. What is the OUTPUT?
3. What TRANSFORMATION happens?
4. WHY does it work?

### Key Functions to Understand:
```python
# From circuit.py
get_circuit_data()  # How does it extract attention patterns?

# From evaluation.py
evaluate_faithfulness()  # How does it compute sufficiency/comprehensiveness?

# From explainer.py
generate_learned_explanation()  # How does Claude use circuit data?
```

---

## Phase 4: Be Able to Explain (Week 5-6)

### Can You Explain to Someone Else?

**RQ1:** "I found that L9H9 is the most important head for IOI because..."

**RQ2:** "My NL explanations are faithful because..."

**RQ3:** "Explanations fail 34% of the time because..."

**Novel:** "No one has done this before because..."

---

## Key Concepts to Master

| Concept | Where to Learn | Where You Used It |
|---------|---------------|-------------------|
| Attention | Karpathy video | `02_ioi_reproduction.ipynb` |
| Circuits | Neel Nanda | `02_ioi_reproduction.ipynb` |
| Activation Patching | ARENA 3.0 | `02_ioi_reproduction.ipynb` |
| Sufficiency | ERASER paper | `evaluation.py` |
| Comprehensiveness | ERASER paper | `evaluation.py` |
| Direct Logit Attribution | TransformerLens docs | `evaluation.py` |

---

## Self-Test Questions

### Basics
- [ ] What is an attention head?
- [ ] What is a residual stream?
- [ ] What does "logit" mean?

### IOI Task
- [ ] What is the IOI task testing?
- [ ] Why is Mary the correct answer in "John gave to Mary"?
- [ ] What do Name Mover heads do?
- [ ] What do S-Inhibition heads do?

### Your Method
- [ ] What is sufficiency?
- [ ] What is comprehensiveness?
- [ ] Why is 100% sufficiency good?
- [ ] Why is 22% comprehensiveness concerning?

### Novel Contribution
- [ ] How does template explanation differ from learned?
- [ ] Why are learned explanations 63% better?
- [ ] What data does Claude receive to generate explanations?

---

## Timeline

| Week | Focus | Goal |
|------|-------|------|
| 1 | Review notebooks | Understand what you built |
| 2-3 | Watch videos | Understand theory |
| 4 | Deep code dive | Understand how it works |
| 5-6 | Practice explaining | Be able to teach it |
| 7+ | Write thesis | Document everything |

---

## Remember

> You built something novel. Now understand it deeply.
> The code is done. The learning starts now.
