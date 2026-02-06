# ⚠️ LIMITATIONS AND DISCLAIMERS

## MSc Thesis: Explainable AI for LLMs via Mechanistic Interpretability
**Author:** Ajay Mahale

---

## 🔴 CRITICAL LIMITATIONS (Must Acknowledge in Thesis)

### 1. MODEL SCOPE

```
LIMITATION: All experiments were conducted on GPT-2 Small (124M parameters).
Results may not generalize to:
- Larger models (GPT-3, GPT-4, Claude, Llama)
- Different architectures (Mamba, RWKV)
- Fine-tuned or instruction-tuned models
```

**Where to state:** Chapter 1 (Introduction), Chapter 6 (Discussion), Chapter 7 (Conclusion)

---

### 2. TASK SCOPE

```
LIMITATION: All experiments focus on the Indirect Object Identification (IOI) task.
Results may not generalize to:
- Other linguistic tasks (sentiment, QA, summarization)
- Mathematical reasoning
- Code generation
- Multi-step reasoning
```

**Where to state:** Chapter 3 (Methodology), Chapter 6 (Discussion)

---

### 3. SAMPLE SIZE

```
LIMITATION: Sample sizes vary across experiments:
- ERASER evaluation: n=50 (canonical)
- Template vs Learned: n=30 (API cost constraint)
- Learned NL generator: n=20 (API cost constraint)

Larger sample sizes may yield different statistical properties.
```

**Where to state:** Chapter 4 (Implementation), each results section

---

### 4. HEURISTIC THRESHOLDS

```
LIMITATION: Comprehensiveness thresholds are HEURISTIC, not statistically derived:
- High comprehensiveness: >25%
- Near threshold: 15-25%
- Low comprehensiveness: <15%

These thresholds were chosen for interpretability.
Different thresholds would yield different failure categorizations.
```

**Where to state:** Chapter 3 (Methodology - explicitly), Chapter 5 (Results - remind)

---

### 5. LLM AS RENDERER

```
CLARIFICATION: The LLM (Claude) is used as a language RENDERER, not an explainer.
- Claude converts mechanistic data to natural language
- Claude does NOT discover or verify the circuit
- Claude's explanations are NOT "ground truth"
- The mechanistic analysis provides ground truth; Claude formats it
```

**Where to state:** Chapter 3 (Methodology), Chapter 4 (Implementation)

---

### 6. COMPREHENSIVENESS GAP

```
FINDING (Not a bug): 22% average comprehensiveness indicates:
- Significant distributed computation exists
- Cited 6 heads are necessary but not fully comprehensive
- This is a PROPERTY of the task, not a failure of the method
```

**Where to state:** Chapter 5 (Results), Chapter 6 (Discussion)

---

## 🟡 METHODOLOGICAL LIMITATIONS

### 7. NO HUMAN EVALUATION

```
LIMITATION: Explanation quality is measured via automated metrics, not human evaluation.
- Quality scoring uses code-based rules (mentions heads, uses percentages, etc.)
- No human subjects rated explanation usefulness
- Future work should include human studies
```

**Where to state:** Chapter 3 (Methodology), Chapter 7 (Future Work)

---

### 8. SINGLE LLM FOR GENERATION

```
LIMITATION: Only Claude (Sonnet) was used for learned explanations.
- Other LLMs (GPT-4, Llama) may produce different quality
- Results are not claims about "LLMs in general"
- API costs prevented multi-LLM comparison
```

**Where to state:** Chapter 4 (Implementation), Chapter 7 (Future Work)

---

### 9. NO ADVERSARIAL TESTING

```
LIMITATION: No adversarial prompts were tested.
- All prompts follow standard IOI format
- Unusual names, non-English names, etc. not tested
- Edge cases may reveal failure modes
```

**Where to state:** Chapter 6 (Discussion), Chapter 7 (Future Work)

---

## 🟢 SCOPE LIMITATIONS (Not Bugs)

### 10. CIRCUIT SELECTION

```
NOTE: The 6-head circuit was selected based on Wang et al. (2022).
- This is established knowledge, not our contribution
- We validate, not discover, the circuit
- Different k values (4, 8, 10 heads) were tested; k=6 is optimal
```

### 11. TEMPLATE SIMPLICITY

```
NOTE: Template explanations are intentionally simple.
- They serve as a BASELINE, not a competitor
- Real-world systems might use more sophisticated templates
- The comparison demonstrates learned > template, not learned = optimal
```

### 12. ENGLISH ONLY

```
NOTE: All experiments are in English.
- IOI task uses English names
- Model is English-trained
- Multilingual generalization not tested
```

---

## 📝 HOW TO WRITE LIMITATIONS SECTION

### Chapter 6: Discussion - Limitations

```
6.4 Limitations

This work has several important limitations that should be considered when 
interpreting our results.

SCOPE LIMITATIONS. All experiments were conducted on GPT-2 Small (124M parameters) 
using the Indirect Object Identification task. While this provides a well-controlled 
experimental setting with known ground truth (Wang et al., 2022), our results may 
not directly generalize to larger models, different architectures, or other tasks. 
Future work should validate these methods on a broader range of models and tasks.

METHODOLOGICAL LIMITATIONS. Our comprehensiveness thresholds (15%, 25%) are heuristic 
values chosen for interpretability, not derived from statistical analysis. Different 
threshold choices would yield different categorizations of failure cases. Additionally, 
explanation quality was evaluated using automated metrics rather than human judgment, 
which may not fully capture usefulness to end users.

SAMPLE SIZE. While our canonical ERASER evaluation uses n=50 prompts, some experiments 
(learned explanation generation, template comparison) use smaller samples (n=20-30) 
due to API cost constraints. Larger samples may reveal additional patterns or alter 
statistical properties of our results.

LLM ROLE. It is important to note that the LLM (Claude) in our pipeline serves as a 
language renderer, converting mechanistic circuit data into natural language—not as 
an explainer that discovers or validates mechanisms. The causal ground truth comes 
from circuit analysis and activation patching, not from the LLM's knowledge or reasoning.
```

---

## ✅ THESIS DEFENSE PREPARATION

### If examiner asks: "Why only GPT-2?"

**Your answer:**
> "GPT-2 Small has well-characterized circuits from prior work, particularly Wang et al.'s 
> IOI circuit. This provides independent ground truth for validation. Testing on larger 
> models is important future work, but establishing rigorous methodology on a known 
> circuit is the appropriate first step."

### If examiner asks: "How do you know this generalizes?"

**Your answer:**
> "I don't claim generalization. This work establishes a methodology and demonstrates 
> it on one well-understood task. Generalization is explicitly listed as a limitation 
> and future work direction. The contribution is the framework, not universal applicability."

### If examiner asks: "Why not human evaluation?"

**Your answer:**
> "Human evaluation is valuable but expensive and introduces subjectivity. My automated 
> metrics provide reproducible, objective measurements. Future work should complement 
> these with human studies, but automated evaluation establishes a baseline that 
> doesn't depend on human variability."

### If examiner asks: "Aren't your thresholds arbitrary?"

**Your answer:**
> "Yes, they are heuristic and I acknowledge this explicitly. The thresholds are chosen 
> for interpretability—15% and 25% provide natural 'low/medium/high' categories. Different 
> thresholds would change the categorization but not the underlying comprehensiveness 
> distribution. The key finding is the 22% average, not the categorical labels."

---

## 📊 LIMITATIONS CHECKLIST

| Limitation | Acknowledged? | Where? |
|------------|---------------|--------|
| GPT-2 only | ⬜ | Ch 1, 6, 7 |
| IOI task only | ⬜ | Ch 3, 6 |
| Sample sizes vary | ⬜ | Ch 4, results |
| Heuristic thresholds | ⬜ | Ch 3, 5 |
| LLM as renderer | ⬜ | Ch 3, 4 |
| No human evaluation | ⬜ | Ch 3, 7 |
| Single LLM (Claude) | ⬜ | Ch 4, 7 |
| English only | ⬜ | Ch 3 |

Check each box as you add these to your thesis document.
