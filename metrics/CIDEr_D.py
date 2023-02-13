import numpy as np
import nltk
from nltk.metrics.distance import edit_distance
from nltk.translate.bleu_score import sentence_bleu


def cide_r(refs, hypo):
    # Compute the sentence level BLEU score for each reference-hypothesis pair
    bleu_scores = [sentence_bleu([ref], hypo) for ref in refs]
    # Compute the mean sentence level BLEU score for the given reference captions and hypothesis
    mean_bleu_score = np.mean(bleu_scores)
    # Compute the log of the mean BLEU scorexl
    log_mean_bleu_score = np.log(mean_bleu_score)
    return log_mean_bleu_score


# Example usage:
references = [["this", "is", "a", "dog"], ["this", "is", "a", "cat"]]
hypothesis = ["this", "is", "a", "dog"]
score = cide_r(references, hypothesis)
print("CIDEr-D score:", score)
