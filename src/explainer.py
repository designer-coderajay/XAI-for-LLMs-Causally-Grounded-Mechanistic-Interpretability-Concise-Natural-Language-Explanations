"""Natural language explanation generation."""

def generate_template_explanation(circuit_data):
    """Template-based explanation (baseline)."""
    return f"The model predicts \"{circuit_data['prediction']}\" because L9H9 and L9H6 attend to it with high attention, copying the indirect object to output position."


def generate_learned_explanation(circuit_data, client):
    """
    NOVEL: LLM-generated explanation from circuit data.
    Requires anthropic client.
    """
    
    user_prompt = f"""You are an AI interpretability expert. Explain this GPT-2 prediction in 1-2 sentences.

INPUT: "{circuit_data['prompt']}"
PREDICTION: "{circuit_data['prediction']}" (confidence: {circuit_data['confidence']})

MECHANISTIC DATA:
- Indirect object: {circuit_data['indirect_object']}
- Subject: {circuit_data['subject']}
- L9H9 attention: {circuit_data['attention'].get('L9H9', {})}
- L9H6 attention: {circuit_data['attention'].get('L9H6', {})}

Rules: Be concise, mention specific heads with percentages, no hedging."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": user_prompt}]
    )
    
    return response.content[0].text
