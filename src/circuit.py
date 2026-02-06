"""Circuit detection and analysis."""

import torch

# IOI Circuit heads (from Wang et al. 2023, reproduced)
our_head_importance = {
    (9, 9): 0.174,   # Name Mover (most important)
    (8, 10): 0.123,  # S-Inhibition
    (7, 3): 0.103,   # S-Inhibition
    (10, 6): 0.089,  # Backup Name Mover
    (9, 6): 0.063,   # Name Mover
    (10, 0): 0.062,  # Backup Name Mover
}

NOT_NAMES = {
    "When", "The", "Then", "There", "They", 
    "This", "That", "What", "Where", "Who", "How"
}

def get_circuit_data(prompt, model):
    """Extract mechanistic circuit data from GPT-2."""
    
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    
    # Get prediction
    pred_token = logits[0, -1].argmax().item()
    prediction = model.to_string([pred_token]).strip()
    confidence = torch.softmax(logits[0, -1], dim=-1)[pred_token].item()
    
    # Find names
    names = []
    for i, tok in enumerate(token_strs):
        tok_clean = tok.strip()
        if (tok_clean and tok_clean[0].isupper() and tok_clean.isalpha() and 
            len(tok_clean) > 1 and tok_clean not in NOT_NAMES):
            if tok_clean not in [n[1] for n in names]:
                names.append((i, tok_clean))
    
    # Get attention data from key heads
    attention_data = {}
    key_heads = [(9, 9), (9, 6), (8, 10), (7, 3)]
    
    for layer, head in key_heads:
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head, -1]
        head_name = f"L{layer}H{head}"
        attention_data[head_name] = {}
        for pos, name in names:
            attention_data[head_name][name] = f"{pattern[pos].item():.1%}"
    
    return {
        "prompt": prompt,
        "prediction": prediction,
        "confidence": f"{confidence:.1%}",
        "confidence_float": confidence,
        "names": [n[1] for n in names],
        "attention": attention_data,
        "indirect_object": names[0][1] if names else None,
        "subject": names[1][1] if len(names) > 1 else None,
    }
