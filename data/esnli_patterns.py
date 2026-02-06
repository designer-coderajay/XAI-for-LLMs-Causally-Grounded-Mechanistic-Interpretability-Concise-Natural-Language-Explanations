"""e-SNLI Explanation Patterns (for NL generation reference)."""

# Key patterns from e-SNLI explanations
ESNLI_PATTERNS = {
    "concise": "8-15 words, 1 sentence",
    "causal_language": ["X is Y", "X does not imply Y", "X cannot be Y"],
    "reference_input": "Mention specific elements from input",
    "no_hedging": "Confident, factual tone - no maybe/perhaps",
}

# e-SNLI style templates adapted for IOI
IOI_TEMPLATES = {
    "high_confidence": "{prediction} is predicted because {head} attends to it ({attention}%), indicating the recipient.",
    "mechanism": "{head} copies '{prediction}' to output, as it identifies the indirect object.",
    "causal": "Ablating {head} drops accuracy by {effect}%, proving it causes the prediction.",
    "contrast": "'{subject}' is suppressed by S-Inhibition heads, allowing '{prediction}' to be output.",
}

# Example e-SNLI explanations (for reference)
ESNLI_EXAMPLES = [
    {
        "premise": "A man selling donuts to a customer during a busy delivery.",
        "hypothesis": "A man is selling donuts.",
        "label": "entailment",
        "explanation": "A man selling donuts is selling donuts."
    },
    {
        "premise": "A smiling costumed woman is holding an umbrella.",
        "hypothesis": "A woman is frowning.",
        "label": "contradiction",
        "explanation": "A smiling woman cannot be frowning at the same time."
    },
]
