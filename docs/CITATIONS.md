# 📚 KEY CITATIONS FOR THESIS

## MSc Thesis: Explainable AI for LLMs via Mechanistic Interpretability
**Author:** Ajay Mahale  
**Institution:** Hochschule Trier  
**Supervisor:** Prof. Dr. Haffner

---

## 🔴 MUST CITE (Core Papers)

### 1. IOI Circuit Discovery (Your Task Foundation)
```bibtex
@article{wang2022interpretability,
  title={Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small},
  author={Wang, Kevin and Variengien, Alexandre and Conmy, Arthur and Shlegeris, Buck and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2211.00593},
  year={2022}
}
```
**Why cite:** This is THE paper for the IOI task. Your circuit heads (L9H9, L8H10, etc.) come from here.

---

### 2. ERASER Benchmark (Your Evaluation Framework)
```bibtex
@inproceedings{deyoung2020eraser,
  title={ERASER: A Benchmark to Evaluate Rationalized NLP Models},
  author={DeYoung, Jay and Jain, Sarthak and Rajani, Nazneen Fatema and Lehman, Eric and Xiong, Caiming and Socher, Richard and Wallace, Byron C},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={4443--4458},
  year={2020}
}
```
**Why cite:** Sufficiency and comprehensiveness metrics come from here.

---

### 3. Mathematical Framework for Circuits (Mechanistic Interp Foundation)
```bibtex
@article{elhage2021mathematical,
  title={A Mathematical Framework for Transformer Circuits},
  author={Elhage, Nelson and Nanda, Neel and Olsson, Catherine and others},
  journal={Transformer Circuits Thread},
  year={2021},
  url={https://transformer-circuits.pub/2021/framework/index.html}
}
```
**Why cite:** Foundation for mechanistic interpretability approach.

---

### 4. Language Models Can Explain Neurons (CLOSEST RELATED WORK)
```bibtex
@article{bills2023language,
  title={Language Models Can Explain Neurons in Language Models},
  author={Bills, Steven and Cammarata, Nick and Mossing, Dan and others},
  journal={OpenAI Blog},
  year={2023},
  url={https://openai.com/research/language-models-can-explain-neurons}
}
```
**Why cite:** ⚠️ CRITICAL - This is your closest related work. You MUST discuss how your work differs:
- They explain individual neurons → You explain circuits
- They use simulation scoring → You use ERASER metrics
- They focus on neuron-level → You focus on task-level (IOI)

---

### 5. Attention is Not Explanation (Why Attention ≠ Causality)
```bibtex
@inproceedings{jain2019attention,
  title={Attention is Not Explanation},
  author={Jain, Sarthak and Wallace, Byron C},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics},
  pages={3543--3556},
  year={2019}
}
```
**Why cite:** Justifies why you use causal interventions (activation patching) instead of just attention weights.

---

### 6. TransformerLens (Your Tool)
```bibtex
@software{nanda2022transformerlens,
  title={TransformerLens},
  author={Nanda, Neel and Bloom, Joseph},
  year={2022},
  url={https://github.com/neelnanda-io/TransformerLens}
}
```
**Why cite:** The library you use for circuit analysis.

---

### 7. Automated Circuit Discovery
```bibtex
@article{conmy2023towards,
  title={Towards Automated Circuit Discovery for Mechanistic Interpretability},
  author={Conmy, Arthur and Mavor-Parker, Augustine and Lynch, Aengus and Heimersheim, Stefan and Garriga-Alonso, Adri{\`a}},
  journal={arXiv preprint arXiv:2304.14997},
  year={2023}
}
```
**Why cite:** Shows automated approaches to circuit discovery (contrast with your approach).

---

## 🟡 SHOULD CITE (Supporting Papers)

### 8. GPT-2 Original Paper
```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Blog},
  year={2019}
}
```

### 9. Attention is All You Need (Transformer Architecture)
```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

### 10. e-SNLI Dataset (Your Format Reference)
```bibtex
@inproceedings{camburu2018snli,
  title={e-SNLI: Natural Language Inference with Natural Language Explanations},
  author={Camburu, Oana-Maria and Rockt{\"a}schel, Tim and Lukasiewicz, Thomas and Blunsom, Phil},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```

### 11. LIME (Alternative Explanation Method)
```bibtex
@inproceedings{ribeiro2016lime,
  title={Why Should I Trust You?: Explaining the Predictions of Any Classifier},
  author={Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={1135--1144},
  year={2016}
}
```

### 12. SHAP (Alternative Explanation Method)
```bibtex
@inproceedings{lundberg2017unified,
  title={A Unified Approach to Interpreting Model Predictions},
  author={Lundberg, Scott M and Lee, Su-In},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

---

## 🟢 OPTIONAL (For Comprehensive Literature Review)

### 13. Induction Heads
```bibtex
@article{olsson2022context,
  title={In-context Learning and Induction Heads},
  author={Olsson, Catherine and Elhage, Nelson and Nanda, Neel and others},
  journal={Transformer Circuits Thread},
  year={2022}
}
```

### 14. Superposition
```bibtex
@article{elhage2022superposition,
  title={Toy Models of Superposition},
  author={Elhage, Nelson and Hume, Tristan and Olsson, Catherine and others},
  journal={Transformer Circuits Thread},
  year={2022}
}
```

### 15. Anthropic's Mechanistic Interpretability
```bibtex
@misc{anthropic2023mechanistic,
  title={Mechanistic Interpretability},
  author={Anthropic},
  year={2023},
  url={https://www.anthropic.com/research#interpretability}
}
```

---

## 📋 RELATED WORK SECTION TEMPLATE

Use this in your thesis Chapter 2:

```
2.X Related Work

The most closely related work to ours is Bills et al. (2023), who demonstrated 
that language models can generate natural language explanations for individual 
neurons in other language models. Their approach uses GPT-4 to explain neurons 
in GPT-2, evaluated via simulation scoring.

Our work differs in several key aspects:
1. GRANULARITY: We explain circuits (multiple coordinated attention heads) 
   rather than individual neurons, providing task-level rather than 
   feature-level explanations.
2. EVALUATION: We adapt ERASER metrics (DeYoung et al., 2020) for faithfulness 
   evaluation, using causal interventions (sufficiency, comprehensiveness) 
   rather than simulation scoring.
3. TASK FOCUS: We ground our work in the well-characterized IOI circuit 
   (Wang et al., 2022), enabling validation against known mechanistic ground truth.

Our work also builds on the mechanistic interpretability framework of Elhage et al. 
(2021) and uses TransformerLens (Nanda, 2022) for circuit analysis. We chose 
causal interventions over attention-based explanations following the findings 
of Jain & Wallace (2019) that attention weights do not reliably indicate 
causal importance.
```

---

## ⚠️ IMPORTANT NOTES

1. **Bills et al. 2023 is your CLOSEST related work** - You must discuss it explicitly
2. **Claim "to our knowledge"** not "first ever"
3. **Acknowledge limitations** - GPT-2 only, IOI task only
4. **Cite TransformerLens** - It's the tool you use
5. **Explain why causal** - Cite Jain & Wallace on attention ≠ causality

---

## 📊 CITATION COUNT SUMMARY

| Category | Count | Papers |
|----------|-------|--------|
| MUST CITE | 7 | Wang, DeYoung, Elhage, Bills, Jain, Nanda, Conmy |
| SHOULD CITE | 5 | Radford, Vaswani, Camburu, Ribeiro, Lundberg |
| OPTIONAL | 3 | Olsson, Elhage (superposition), Anthropic |
| **TOTAL** | **15** | |

For an MSc thesis, citing **10-15 papers** in your experimental/implementation chapters is appropriate.
