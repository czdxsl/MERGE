from nltk.util import ngrams
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np


def calculate_bleu(references, hypothesis, n=4, smooth=False):
    '''
    :param references: a list of reference sentences, where each reference is a list of words
    :param hypothesis: a list of words representing the hypothesis sentence
    :param n: the maximum order of n-grams to use in the BLEU calculation
    :param smooth: a flag indicating whether to apply smoothing to the BLEU calculation
    :return: the BLEU-n score as a float
    '''
    smoothie = SmoothingFunction().method4
    weights = np.array([1 / n] * n)
    references = [[reference.split()] for reference in references]

    clipped_counts = np.zeros(n)
    total_counts = np.zeros(n)
    hypothesis_length = len(hypothesis)
    for reference in references:
        reference_length = len(reference[0])
        diff = abs(reference_length - hypothesis_length)
        for i in range(n):
            if i < diff:
                continue
            reference_ngrams = ngrams(reference[0], i + 1)
            hypothesis_ngrams = ngrams(hypothesis, i + 1)
            counts = np.zeros((len(reference_ngrams),))
            for j, hypothesis_ngram in enumerate(hypothesis_ngrams):
                for k, reference_ngram in enumerate(reference_ngrams):
                    if hypothesis_ngram == reference_ngram:
                        counts[k] = 1
                        break
            clipped_counts[i] += np.sum(np.minimum(counts, 1))
            total_counts[i] += len(hypothesis_ngrams)

    if smooth:
        clipped_counts = clipped_counts + 1
        total_counts = total_counts + 1

    # brevity penalty
    if hypothesis_length > reference_length:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - reference_length / hypothesis_length)

    score = brevity_penalty * np.exp(np.sum(np.log(clipped_counts / total_counts)) / n)
    return score
