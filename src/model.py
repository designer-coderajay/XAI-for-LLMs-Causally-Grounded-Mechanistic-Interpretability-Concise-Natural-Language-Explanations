"""Model loading utilities."""

import torch
from transformer_lens import HookedTransformer

def load_model(model_name="gpt2-small", device="cuda"):
    """Load GPT-2 model with TransformerLens."""
    model = HookedTransformer.from_pretrained(model_name)
    if device == "cuda" and torch.cuda.is_available():
        model.cuda()
    return model

# Constants
NOT_NAMES = {
    "When", "The", "Then", "There", "They", 
    "This", "That", "What", "Where", "Who", "How"
}
