"""ERASER-style faithfulness evaluation."""

import torch
import numpy as np

def evaluate_faithfulness(prompt, model, head_importance, k=6, NOT_NAMES=None):
    """
    Evaluate sufficiency and comprehensiveness (ERASER metrics).
    
    Returns:
        dict with 'sufficiency' and 'comprehensiveness' scores
    """
    if NOT_NAMES is None:
        NOT_NAMES = {"When", "The", "Then", "There", "They", 
                     "This", "That", "What", "Where", "Who", "How"}
    
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)
    
    # Find names
    names = []
    for i, tok in enumerate(token_strs):
        tok_clean = tok.strip()
        if (tok_clean and tok_clean[0].isupper() and tok_clean.isalpha() and 
            len(tok_clean) > 1 and tok_clean not in NOT_NAMES):
            if tok_clean not in [n[1] for n in names]:
                names.append((i, tok_clean))
    
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
    
    # Top k heads
    top_heads = sorted(head_importance.items(), key=lambda x: -x[1])[:k]
    top_head_set = set([h[0] for h in top_heads])
    
    # Sufficiency
    cited_contribution = sum(head_contributions.get(h, 0) for h in top_head_set)
    sufficiency = cited_contribution / clean_diff if clean_diff > 0 else 0
    
    # Comprehensiveness
    def remove_important(activation, hook):
        layer = int(hook.name.split(".")[1])
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
        "sufficiency": max(0, min(1, sufficiency)),
        "comprehensiveness": max(0, min(1, comprehensiveness)),
    }
