from transformers import AutoTokenizer, AutoModel
import numpy as np


def bert_score(generated_text, reference_text, model_type="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)

    generated_tokens = tokenizer.tokenize(generated_text)
    reference_tokens = tokenizer.tokenize(reference_text)

    # Get BERT embeddings
    generated_embeddings = model(tokenizer.encode(generated_text, return_tensors="pt"))[
        0].last_hidden_state.detach().numpy()
    reference_embeddings = model(tokenizer.encode(reference_text, return_tensors="pt"))[
        0].last_hidden_state.detach().numpy()

    # Compute BERTScore
    p = np.exp(-np.sum((generated_embeddings - reference_embeddings) ** 2, axis=1))
    p /= np.sum(p)
    reference_len = len(reference_tokens)
    generated_len = len(generated_tokens)

    # Compute the F1 scores
    f1_scores = 2 * p * (generated_len / (generated_len + reference_len))
    f1_scores = np.maximum(f1_scores - 1, 0)

    return np.mean(f1_scores)
