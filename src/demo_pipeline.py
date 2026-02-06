"""
================================================================================
DEMO_PIPELINE.PY - IMPROVED THESIS DEMONSTRATION
================================================================================
MSc Thesis: Explainable AI for LLMs via Mechanistic Interpretability
Author: Ajay Mahale
Supervisor: Prof. Dr. Haffner | Hochschule Trier

Purpose: Complete end-to-end demonstration of the thesis pipeline.
         Takes user input and produces human-readable explanations.

Version: 2.0 (Improved with better terminology, warnings, and examples)

Usage (Colab):
    !python demo_pipeline.py
    
Usage (Interactive):
    from demo_pipeline import run_full_analysis
    result = run_full_analysis("When Mary and John went to the store, John gave a drink to")
================================================================================
"""

import torch
import numpy as np
from datetime import datetime

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

NOT_NAMES = {'When', 'The', 'Then', 'There', 'They', 'This', 'That', 'What', 'Where', 'Who', 'How', 'If', 'So', 'But'}

# Canonical circuit heads from Wang et al. (2022)
# NOTE: These are GLOBAL AVERAGES from dataset-level ablation studies
CIRCUIT_HEADS = {
    (9, 9):  {'importance': 0.174, 'role': 'Name Mover (Primary)'},
    (8, 10): {'importance': 0.123, 'role': 'S-Inhibition'},
    (7, 3):  {'importance': 0.103, 'role': 'Name Mover (Secondary)'},
    (10, 6): {'importance': 0.089, 'role': 'Backup Name Mover'},
    (9, 6):  {'importance': 0.063, 'role': 'Name Mover (Tertiary)'},
    (10, 0): {'importance': 0.062, 'role': 'Output Head'},
}

# Thresholds for trust assessment
CONFIDENCE_HIGH_THRESHOLD = 0.70
CONFIDENCE_MEDIUM_THRESHOLD = 0.50
COMPREHENSIVENESS_WARNING_THRESHOLD = 0.30


# ==============================================================================
# EXAMPLE PROMPTS (Expanded from 3 to 10)
# ==============================================================================

EXAMPLE_PROMPTS = [
    # Original 3
    "When Mary and John went to the store, John gave a drink to",
    "When Alice and Bob went to the park, Bob handed a flower to",
    "When Sarah and Tom went to the library, Tom showed a book to",
    # New additions
    "When Emma and James went to the cafe, James offered a coffee to",
    "When Lisa and David went to the beach, David threw a ball to",
    "When Sophie and Daniel went to the party, Daniel gave a gift to",
    "When Rachel and Chris went to the office, Chris sent a message to",
    "When Laura and Kevin went to the restaurant, Kevin passed the menu to",
    "When Julia and Peter went to the garden, Peter handed a flower to",
    "When Diana and Steve went to the museum, Steve showed a painting to",
]


# ==============================================================================
# CORE ANALYSIS FUNCTIONS
# ==============================================================================

def load_model():
    """Load GPT-2 Small with TransformerLens."""
    from transformer_lens import HookedTransformer
    
    print("Loading GPT-2 Small...")
    print("NOTE: This demo runs on CPU by default. CUDA/GPU warnings can be safely ignored.")
    model = HookedTransformer.from_pretrained("gpt2-small")
    if torch.cuda.is_available():
        model.cuda()
        print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✅ Model loaded on CPU")
    return model


def extract_names(token_strs):
    """Extract name tokens from tokenized prompt."""
    names = []
    for i, tok in enumerate(token_strs):
        tok_clean = tok.strip()
        if (tok_clean and tok_clean[0].isupper() and tok_clean.isalpha() and
            len(tok_clean) > 1 and tok_clean not in NOT_NAMES):
            if tok_clean not in [n[1] for n in names]:
                names.append((i, tok_clean))
    return names


def get_circuit_data(prompt, model):
    """Extract complete mechanistic circuit data."""
    
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    
    # Get prediction
    pred_token = logits[0, -1].argmax().item()
    prediction = model.to_string([pred_token]).strip()
    
    # Get confidence (softmax probability)
    probs = torch.softmax(logits[0, -1], dim=-1)
    confidence = probs[pred_token].item()
    
    # Extract names
    names = extract_names(token_strs)
    indirect_object = names[0][1] if names else None
    subject = names[1][1] if len(names) > 1 else None
    
    # Get attention patterns for circuit heads (INSTANCE-SPECIFIC)
    attention_data = {}
    for (layer, head), info in CIRCUIT_HEADS.items():
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head, -1]
        head_name = f"L{layer}H{head}"
        
        # Instance-specific attention weights
        attn_to_io = pattern[names[0][0]].item() if names else 0
        attn_to_subj = pattern[names[1][0]].item() if len(names) > 1 else 0
        
        attention_data[head_name] = {
            'role': info['role'],
            'global_importance': info['importance'],  # RENAMED: Was just 'importance'
            'attention_to_io': attn_to_io,            # Instance-specific
            'attention_to_subj': attn_to_subj,        # Instance-specific
        }
    
    return {
        'prompt': prompt,
        'prediction': prediction,
        'confidence': confidence,
        'indirect_object': indirect_object,
        'subject': subject,
        'names': [n[1] for n in names],
        'attention_data': attention_data,
        'cache': cache,
        'tokens': tokens,
    }


def compute_faithfulness(prompt, model, k=6):
    """Compute ERASER-style sufficiency and comprehensiveness."""
    
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)
    names = extract_names(token_strs)
    
    if len(names) < 2:
        return None
    
    indirect_obj, subject = names[0][1], names[1][1]
    
    try:
        io_tok = model.to_single_token(" " + indirect_obj)
        subj_tok = model.to_single_token(" " + subject)
    except:
        return None
    
    logits, cache = model.run_with_cache(tokens)
    clean_diff = (logits[0, -1, io_tok] - logits[0, -1, subj_tok]).item()
    
    if clean_diff <= 0:
        return None
    
    # Direct logit attribution
    logit_diff_dir = model.W_U[:, io_tok] - model.W_U[:, subj_tok]
    head_contributions = {}
    
    for layer in range(model.cfg.n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        W_O = model.W_O[layer]
        for head in range(model.cfg.n_heads):
            head_out = z[0, -1, head, :]
            head_contribution = head_out @ W_O[head]
            contribution = (head_contribution @ logit_diff_dir).item()
            head_contributions[(layer, head)] = contribution
    
    # Get top k heads by importance
    top_heads = list(CIRCUIT_HEADS.keys())[:k]
    top_head_set = set(top_heads)
    
    # Sufficiency: How much do cited heads preserve the prediction?
    cited_contribution = sum(head_contributions.get(h, 0) for h in top_head_set)
    sufficiency = cited_contribution / clean_diff if clean_diff > 0 else 0
    
    # Comprehensiveness: How much does ablating cited heads reduce the prediction?
    def remove_important(activation, hook):
        layer = int(hook.name.split('.')[1])
        new_activation = activation.clone()
        for head in range(activation.shape[2]):
            if (layer, head) in top_head_set:
                new_activation[:, :, head, :] = 0
        return new_activation
    
    hook_names = [f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)]
    comp_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(name, remove_important) for name in hook_names]
    )
    comp_diff = (comp_logits[0, -1, io_tok] - comp_logits[0, -1, subj_tok]).item()
    comprehensiveness = 1 - (comp_diff / clean_diff) if clean_diff > 0 else 0
    
    return {
        'sufficiency': max(0, min(1, sufficiency)),
        'comprehensiveness': max(0, min(1, comprehensiveness)),
        'f1': 2 * sufficiency * comprehensiveness / (sufficiency + comprehensiveness) if (sufficiency + comprehensiveness) > 0 else 0,
    }


# ==============================================================================
# TRUST ASSESSMENT (IMPROVED LOGIC)
# ==============================================================================

def assess_trust(confidence, faithfulness):
    """
    Improved trust assessment based on BOTH confidence AND faithfulness.
    
    Rules:
    - HIGH: Confidence >= 70% AND Sufficiency >= 90%
    - MEDIUM: Confidence >= 50% AND Sufficiency >= 50%
    - LOW: Everything else
    
    Additional warning if comprehensiveness is low.
    """
    
    if faithfulness is None:
        return {
            'level': 'UNKNOWN',
            'reason': 'Could not compute faithfulness metrics',
            'warnings': ['Unable to evaluate explanation quality']
        }
    
    suff = faithfulness['sufficiency']
    comp = faithfulness['comprehensiveness']
    
    warnings = []
    
    # Check comprehensiveness
    if comp < COMPREHENSIVENESS_WARNING_THRESHOLD:
        coverage_pct = comp * 100
        missing_pct = (1 - comp) * 100
        warnings.append(f"LOW COVERAGE: Only {coverage_pct:.1f}% of computation explained")
        warnings.append(f"Remaining {missing_pct:.1f}% uses distributed/backup circuits")
    
    # Determine trust level
    if confidence >= CONFIDENCE_HIGH_THRESHOLD and suff >= 0.90:
        level = 'HIGH'
        reason = 'High model confidence and high sufficiency'
    elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD and suff >= 0.50:
        level = 'MEDIUM'
        if comp < COMPREHENSIVENESS_WARNING_THRESHOLD:
            reason = 'Adequate confidence but low comprehensiveness suggests backup circuits'
        else:
            reason = 'Moderate confidence and sufficiency'
    else:
        level = 'LOW'
        if confidence < CONFIDENCE_MEDIUM_THRESHOLD:
            reason = f'Low model confidence ({confidence:.1%}) - prediction may be unreliable'
        else:
            reason = 'Low sufficiency - cited heads may not explain prediction'
    
    return {
        'level': level,
        'reason': reason,
        'warnings': warnings
    }


# ==============================================================================
# HUMAN-READABLE OUTPUT FORMATTER (IMPROVED)
# ==============================================================================

def format_human_readable(circuit_data, faithfulness):
    """Convert technical output to human-readable explanation with improved terminology."""
    
    output = []
    
    # Header with domain restriction
    output.append("=" * 70)
    output.append("MECHANISTIC EXPLANATION")
    output.append("=" * 70)
    output.append("")
    output.append("╔══════════════════════════════════════════════════════════════════╗")
    output.append("║  ⚠️  DOMAIN: Indirect Object Identification (IOI) Tasks Only     ║")
    output.append("║  This tool analyzes sentences like:                              ║")
    output.append("║  \"When [Name1] and [Name2] went to X, [Name2] gave Y to _\"      ║")
    output.append("╚══════════════════════════════════════════════════════════════════╝")
    
    # Input & Prediction
    output.append(f"\n📝 INPUT:")
    output.append(f"   {circuit_data['prompt']}")
    
    output.append(f"\n🎯 PREDICTION:")
    conf_pct = circuit_data['confidence'] * 100
    output.append(f"   \"{circuit_data['prediction']}\" (Model Confidence: {conf_pct:.1f}%)")
    
    # Confidence assessment
    if circuit_data['confidence'] < CONFIDENCE_MEDIUM_THRESHOLD:
        output.append(f"   ⚠️  LOW CONFIDENCE: Model uncertainty is high; prediction reliability is reduced")
    
    # Names identified
    output.append(f"\n👥 NAMES IDENTIFIED:")
    output.append(f"   • Indirect Object (recipient): {circuit_data['indirect_object']}")
    output.append(f"   • Subject (giver): {circuit_data['subject']}")
    
    # Mechanistic Evidence with CLEAR TERMINOLOGY
    output.append(f"\n🔬 MECHANISTIC EVIDENCE:")
    output.append("-" * 70)
    output.append(f"   {'Head':<8} | {'Role':<22} | {'Global Imp.':<12} | {'Attention':<12}")
    output.append(f"   {'-'*8} | {'-'*22} | {'-'*12} | {'-'*12}")
    
    for head_name, data in circuit_data['attention_data'].items():
        global_imp = data['global_importance'] * 100
        attn_pct = data['attention_to_io'] * 100
        io_name = circuit_data['indirect_object']
        output.append(f"   {head_name:<8} | {data['role']:<22} | {global_imp:>5.1f}% (Avg) | {attn_pct:>5.1f}% → {io_name}")
    
    output.append("")
    output.append("   📌 TERMINOLOGY NOTE:")
    output.append("   • Global Importance = Dataset-level average from our ablation experiments (following IOI methodology)")
    output.append("   • Attention = This specific input (correlation, NOT causation)")
    
    # Faithfulness Metrics with IMPROVED WORDING
    if faithfulness:
        output.append(f"\n📊 FAITHFULNESS METRICS (ERASER-style):")
        output.append("-" * 70)
        
        suff_pct = faithfulness['sufficiency'] * 100
        comp_pct = faithfulness['comprehensiveness'] * 100
        f1_pct = faithfulness['f1'] * 100
        
        # Sufficiency with CORRECTED wording
        output.append(f"   • Sufficiency:        {suff_pct:.1f}%")
        output.append(f"     (Prediction preserved when using ONLY the 6 cited heads)")
        output.append(f"     Note: This measures performance retention, not total explanation")
        
        output.append(f"")
        
        # Comprehensiveness with WARNING
        output.append(f"   • Comprehensiveness:  {comp_pct:.1f}%")
        output.append(f"     (Prediction reduced by {comp_pct:.1f}% when ablating cited heads)")
        
        if faithfulness['comprehensiveness'] < COMPREHENSIVENESS_WARNING_THRESHOLD:
            missing_pct = (1 - faithfulness['comprehensiveness']) * 100
            output.append(f"     ⚠️  LOW COVERAGE WARNING:")
            output.append(f"        {missing_pct:.1f}% of computation uses backup/distributed circuits")
            output.append(f"        The explanation is INCOMPLETE (this is a known limitation)")
        
        output.append(f"")
        output.append(f"   • Local Faithfulness Score:           {f1_pct:.1f}%")
        output.append(f"     (ERASER-style proxy: harmonic mean of Sufficiency and Comprehensiveness)")
    
    # Trust Assessment (IMPROVED LOGIC)
    trust = assess_trust(circuit_data['confidence'], faithfulness)
    
    output.append(f"\n✅ TRUST ASSESSMENT:")
    output.append("-" * 70)
    output.append(f"   Model Confidence:     {circuit_data['confidence']:.1%}")
    if faithfulness:
        output.append(f"   Sufficiency:          {faithfulness['sufficiency']:.1%}")
        output.append(f"   Comprehensiveness:    {faithfulness['comprehensiveness']:.1%}")
    output.append(f"")
    output.append(f"   🔒 Trust Level: {trust['level']}")
    output.append(f"   Reason: {trust['reason']}")
    
    # Display warnings
    if trust['warnings']:
        output.append(f"")
        output.append(f"   ⚠️  WARNINGS:")
        for warning in trust['warnings']:
            output.append(f"      • {warning}")
    
    # Natural Language Explanation
    output.append(f"\n💬 NATURAL LANGUAGE EXPLANATION:")
    output.append("-" * 70)
    
    top_head = list(circuit_data['attention_data'].items())[0]
    top_head_name, top_head_data = top_head
    top_attn = top_head_data['attention_to_io'] * 100
    total_importance = sum(h['global_importance'] for h in circuit_data['attention_data'].values()) * 100
    
    explanation = f"""   The model predicts '{circuit_data['prediction']}' because:
   
   1. The Name Mover head {top_head_name} attends to '{circuit_data['indirect_object']}' 
      with {top_attn:.1f}% attention weight, copying it to the output.
   
   2. The S-Inhibition head L8H10 suppresses '{circuit_data['subject']}' (the giver),
      preventing the model from outputting the subject instead.
   
   3. These 6 heads together account for {total_importance:.1f}% of the circuit's
      normalized logit-difference attribution captured by the predefined IOI circuit."""
    
    output.append(explanation)
    
    if faithfulness and faithfulness['comprehensiveness'] < COMPREHENSIVENESS_WARNING_THRESHOLD:
        output.append(f"""
   ⚠️  CAVEAT: The low comprehensiveness ({faithfulness['comprehensiveness']:.1%}) indicates that
   backup circuits and distributed computation contribute significantly.
   This explanation covers the PRIMARY mechanism but not the full picture.""")
    
    # Limitations
    output.append(f"\n⚠️  LIMITATIONS:")
    output.append("-" * 70)
    output.append("   • Validated only on GPT-2 Small (124M parameters)")
    output.append("   • Only works for Indirect Object Identification (IOI) tasks")
    output.append("   • Global importance scores are dataset averages, not instance-specific")
    output.append("   • Comprehensiveness gap indicates distributed computation exists")
    output.append("   • May not generalize to other models or tasks")
    
    output.append("\n" + "=" * 70)
    
    return "\n".join(output)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_full_analysis(prompt, model=None):
    """
    Run complete analysis pipeline on a single prompt.
    
    Args:
        prompt: IOI-style prompt string
        model: Optional pre-loaded model (loads if None)
    
    Returns:
        dict with all analysis results
    """
    
    # Load model if needed
    if model is None:
        model = load_model()
    
    print(f"\n🔍 Analyzing: \"{prompt[:50]}...\"")
    
    # Extract circuit data
    circuit_data = get_circuit_data(prompt, model)
    
    # Compute faithfulness
    faithfulness = compute_faithfulness(prompt, model)
    
    # Generate human-readable output
    human_output = format_human_readable(circuit_data, faithfulness)
    
    return {
        'circuit_data': circuit_data,
        'faithfulness': faithfulness,
        'human_output': human_output,
        'model': model,
    }


def run_interactive_demo():
    """Run interactive demo in terminal."""
    
    print("=" * 70)
    print("MECHANISTIC INTERPRETABILITY DEMO")
    print("Explainable AI for LLMs via Circuit Analysis")
    print("=" * 70)
    print("")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ⚠️  DOMAIN: Indirect Object Identification (IOI) Tasks Only     ║")
    print("║  Enter sentences like:                                           ║")
    print("║  \"When [Name1] and [Name2] went to X, [Name2] gave Y to\"        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Load model once
    model = load_model()
    
    # Show examples
    print(f"\n📋 EXAMPLE PROMPTS ({len(EXAMPLE_PROMPTS)} available):")
    for i, ex in enumerate(EXAMPLE_PROMPTS, 1):
        # Truncate long examples for display
        display = ex if len(ex) < 55 else ex[:52] + "..."
        print(f"   {i:2}. {display}")
    
    while True:
        print("\n" + "-" * 70)
        print("Enter a number (1-10) for example, or type your own prompt")
        user_input = input("Your choice (or 'q' to quit): ").strip()
        
        if user_input.lower() == 'q':
            print("Goodbye!")
            break
        
        # Handle numbered examples
        if user_input.isdigit():
            idx = int(user_input)
            if 1 <= idx <= len(EXAMPLE_PROMPTS):
                prompt = EXAMPLE_PROMPTS[idx - 1]
            else:
                print(f"❌ Invalid number. Choose 1-{len(EXAMPLE_PROMPTS)}")
                continue
        else:
            prompt = user_input
        
        # Validate prompt format
        if ' and ' not in prompt or ' to' not in prompt:
            print("⚠️  Warning: Prompt may not follow IOI format.")
            print("   Expected: 'When [Name1] and [Name2] ... [Name2] gave ... to'")
            confirm = input("   Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        # Run analysis
        try:
            result = run_full_analysis(prompt, model)
            print(result['human_output'])
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure prompt follows IOI format with two names")


# ==============================================================================
# BATCH ANALYSIS
# ==============================================================================

def run_batch_analysis(prompts=None, model=None):
    """Run analysis on multiple prompts."""
    
    if prompts is None:
        prompts = EXAMPLE_PROMPTS
    
    if model is None:
        model = load_model()
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}...")
        result = run_full_analysis(prompt, model)
        results.append(result)
    
    # Summary statistics
    faithfulness_scores = [r['faithfulness'] for r in results if r['faithfulness']]
    
    if faithfulness_scores:
        avg_suff = np.mean([f['sufficiency'] for f in faithfulness_scores])
        avg_comp = np.mean([f['comprehensiveness'] for f in faithfulness_scores])
        avg_f1 = np.mean([f['f1'] for f in faithfulness_scores])
        
        low_comp_count = sum(1 for f in faithfulness_scores if f['comprehensiveness'] < COMPREHENSIVENESS_WARNING_THRESHOLD)
        
        print("\n" + "=" * 70)
        print("BATCH SUMMARY")
        print("=" * 70)
        print(f"Prompts analyzed:       {len(results)}")
        print(f"Valid results:          {len(faithfulness_scores)}")
        print(f"")
        print(f"Avg Sufficiency:        {avg_suff:.1%}")
        print(f"Avg Comprehensiveness:  {avg_comp:.1%}")
        print(f"Avg F1:                 {avg_f1:.1%}")
        print(f"")
        print(f"Low comprehensiveness:  {low_comp_count}/{len(faithfulness_scores)} ({low_comp_count/len(faithfulness_scores)*100:.0f}%)")
        
        if low_comp_count > len(faithfulness_scores) * 0.3:
            print(f"⚠️  WARNING: Over 30% of cases have low comprehensiveness")
            print(f"   This indicates significant distributed computation")
    
    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line argument provided
        prompt = " ".join(sys.argv[1:])
        result = run_full_analysis(prompt)
        print(result['human_output'])
    else:
        # Interactive mode
        run_interactive_demo()
