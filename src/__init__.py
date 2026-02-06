"""
Thesis: Explainable AI for LLMs
Causally Grounded Mechanistic Interpretability and NL Explanations
Author: Ajay
Institution: Hochschule Trier
"""

from .model import load_model
from .circuit import get_circuit_data, our_head_importance
from .explainer import generate_template_explanation, generate_learned_explanation
from .evaluation import evaluate_faithfulness
