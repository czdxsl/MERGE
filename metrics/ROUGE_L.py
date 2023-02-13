def rouge_l(reference, candidate):
    reference = [w for w in reference if w != ' ']
    candidate = [w for w in candidate if w != ' ']
    m = len(reference)
    n = len(candidate)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == candidate[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
    return max_len / m
