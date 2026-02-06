"""
================================================================================
00_REPRODUCIBILITY_CONFIG.PY
================================================================================
MSc Thesis: Explainable AI for LLMs via Mechanistic Interpretability
Author: Ajay Mahale
Supervisor: Prof. Dr. Haffner | Hochschule Trier

Purpose: Shared configuration for reproducibility across all notebooks.
         Import this file at the start of every notebook.

Usage:
    from reproducibility_config import setup_reproducibility, CANONICAL_RESULTS
================================================================================
"""

import torch
import numpy as np
import random
import sys
from datetime import datetime

# ==============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# ==============================================================================

RANDOM_SEED = 42

def setup_reproducibility(seed=RANDOM_SEED):
    """
    Set all random seeds for reproducible experiments.
    Call this at the start of every notebook.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic operations (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Reproducibility seed set: {seed}")
    return seed


def log_environment():
    """
    Log library versions for reproducibility.
    Call this at the start of every notebook.
    """
    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"Python:          {sys.version.split()[0]}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version:    {torch.version.cuda}")
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    
    try:
        import transformer_lens
        print(f"TransformerLens: {transformer_lens.__version__}")
    except:
        print("TransformerLens: Not installed")
    
    try:
        import anthropic
        print(f"Anthropic:       {anthropic.__version__}")
    except:
        print("Anthropic:       Not installed")
    
    print(f"Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


# ==============================================================================
# CANONICAL RESULTS (Verified Ground Truth)
# ==============================================================================

CANONICAL_RESULTS = {
    # Notebook 02: IOI Circuit (6 heads, 61.4% total coverage)
    'circuit_heads': {
        'L9H9': 0.174,   # Name Mover (highest)
        'L8H10': 0.123,  # S-Inhibition
        'L7H3': 0.103,
        'L10H6': 0.089,
        'L9H6': 0.063,
        'L10H0': 0.062,
    },
    'total_coverage': 0.614,
    'n_circuit_heads': 6,
    'model_accuracy': 1.00,
    
    # Notebook 05: ERASER Metrics (n=50, CANONICAL)
    'canonical_n': 50,
    'sufficiency_mean': 1.00,
    'sufficiency_std': 0.00,
    'comprehensiveness_mean': 0.22,
    'comprehensiveness_std': 0.173,
    'f1_score': 0.36,
    
    # Baselines
    'attention_f1': 0.206,
    'random_f1': 0.328,
    'improvement_vs_attention': 0.75,  # +75%
    
    # Notebook 06: Failure Analysis
    'low_comp_count': 17,
    'low_comp_pct': 0.34,
    'near_thresh_count': 15,
    'near_thresh_pct': 0.30,
    'high_comp_count': 18,
    'high_comp_pct': 0.36,
    'confidence_correlation': 0.009,
    
    # Notebook 08/09: Template vs Learned
    'template_quality': 0.60,
    'learned_quality': 0.98,
    'quality_improvement': 0.63,  # +63%
    'comparison_n': 30,
}


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

EXPERIMENT_CONFIG = {
    # Model
    'model_name': 'gpt2-small',
    'n_layers': 12,
    'n_heads': 12,
    'd_model': 768,
    
    # Dataset
    'n_name_pairs': 25,
    'n_templates': 2,
    'total_prompts': 50,  # 25 × 2
    
    # ERASER evaluation
    'k_heads': 6,  # Number of heads to cite in explanations
    
    # Thresholds (HEURISTIC - explicitly acknowledged)
    'high_comp_threshold': 0.25,  # >25% = high comprehensiveness
    'low_comp_threshold': 0.15,   # <15% = low comprehensiveness
    
    # API (for learned explanations)
    'llm_model': 'claude-sonnet-4-20250514',
    'max_tokens': 100,
}


# ==============================================================================
# DISCLAIMERS AND LIMITATIONS
# ==============================================================================

DISCLAIMERS = {
    'heuristic_thresholds': """
    NOTE: The comprehensiveness thresholds (15%, 25%) are HEURISTIC values 
    chosen for interpretability. They are not derived from statistical analysis.
    Different thresholds may yield different categorizations.
    """,
    
    'generalization': """
    LIMITATION: Results are validated only on GPT-2 Small with the IOI task.
    Generalization to larger models (GPT-4, Claude) or different tasks 
    requires further investigation.
    """,
    
    'llm_renderer': """
    CLARIFICATION: The LLM (Claude) is used as a language RENDERER to convert
    mechanistic circuit data into natural language. It is NOT the model being
    explained, and it does NOT generate "ground truth" explanations.
    """,
    
    'sample_size': """
    NOTE: Canonical ERASER evaluation uses n=50 prompts. Some experiments 
    (Notebooks 08, 09) use smaller samples (n=20, n=30) due to API costs.
    These are clearly labeled as preliminary/comparison studies.
    """,
}


# ==============================================================================
# NOT NAMES (Common words that look like names but aren't)
# ==============================================================================

NOT_NAMES = {
    'When', 'The', 'Then', 'There', 'They', 'This', 'That', 
    'What', 'Where', 'Who', 'How', 'Why', 'If', 'So', 'But'
}


# ==============================================================================
# NAME PAIRS FOR IOI DATASET
# ==============================================================================

NAME_PAIRS = [
    ("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Emma", "James"),
    ("Lisa", "David"), ("Anna", "Michael"), ("Sophie", "Daniel"), ("Rachel", "Chris"),
    ("Laura", "Kevin"), ("Julia", "Peter"), ("Diana", "Steve"), ("Helen", "Mark"),
    ("Grace", "Paul"), ("Claire", "Andrew"), ("Emily", "Ryan"), ("Olivia", "Nathan"),
    ("Mia", "Lucas"), ("Ella", "Henry"), ("Lily", "Jack"), ("Zoe", "Sam"),
    ("Kate", "Ben"), ("Amy", "Luke"), ("Nina", "Max"), ("Eva", "Leo"), ("Iris", "Adam"),
]

IOI_TEMPLATES = [
    "When {name1} and {name2} went to the store, {name2} gave a drink to",
    "When {name1} and {name2} went to the park, {name2} handed a flower to",
]


def generate_ioi_prompts(n_pairs=None):
    """Generate IOI prompts from name pairs and templates."""
    pairs = NAME_PAIRS[:n_pairs] if n_pairs else NAME_PAIRS
    prompts = []
    for name1, name2 in pairs:
        for template in IOI_TEMPLATES:
            prompts.append(template.format(name1=name1, name2=name2))
    return prompts


# ==============================================================================
# QUICK SETUP FUNCTION
# ==============================================================================

def quick_setup(seed=RANDOM_SEED, log_env=True):
    """
    Quick setup for notebooks. Call at the start of Cell 2.
    
    Usage:
        from reproducibility_config import quick_setup
        quick_setup()
    """
    setup_reproducibility(seed)
    if log_env:
        log_environment()
    return CANONICAL_RESULTS, EXPERIMENT_CONFIG


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("Testing reproducibility config...")
    setup_reproducibility()
    log_environment()
    
    print(f"\nCanonical Results:")
    for key, val in CANONICAL_RESULTS.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        elif isinstance(val, dict):
            print(f"  {key}: {len(val)} items")
        else:
            print(f"  {key}: {val}")
    
    print(f"\nGenerated {len(generate_ioi_prompts())} IOI prompts")
    print("\n✅ Config test passed!")
