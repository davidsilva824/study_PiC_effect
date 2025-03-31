### file to test code in general

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("PatrickHaller/hgrn2_pile_100m_distill_babylm")
model = AutoModelForCausalLM.from_pretrained("PatrickHaller/hgrn2_pile_100m_distill_babylm", trust_remote_code=True)
model.eval()

def compute_surprisal(text, model, tokenizer):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Shift input and target for next-token prediction
    with torch.no_grad():
        logits = model(input_ids).logits

    # Get probabilities of actual next tokens
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute token-level surprisals: -log(P(w_t | context))
    surprisals = []
    for i in range(1, input_ids.size(1)):
        token_id = input_ids[0, i]
        context_log_probs = log_probs[0, i-1]
        surprisal = -context_log_probs[token_id].item()
        token = tokenizer.decode([token_id])
        surprisals.append((token, surprisal))

    return surprisals

sentence = "The cat chased the rat."
surprisals = compute_surprisal(sentence, model, tokenizer)

for token, s in surprisals:
    print(f"Token: {token!r}, Surprisal: {s:.4f}")
