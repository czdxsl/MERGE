import numpy as np
from nltk.metrics import spice

def calc_spice(references, hypothesis):
    spice_scores = []
    for ref in references:
        score = spice(hypothesis, ref)
        spice_scores.append(score)
    return np.mean(spice_scores)
