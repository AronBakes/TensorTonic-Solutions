import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    n = len(rater1)
    
    # Build frequency maps manually
    freq1 = {}
    for label in rater1:
        freq1[label] = freq1.get(label, 0) + 1
        
    freq2 = {}
    for label in rater2:
        freq2[label] = freq2.get(label, 0) + 1
    
    # Observed agreement (p_o)
    agreements = 0
    for i in range(n):
        if rater1[i] == rater2[i]:
            agreements += 1
    p_o = agreements / n
    
    # Chance agreement (p_e)
    p_e = 0.0
    
    # Get all unique labels from both raters
    all_labels = set(freq1.keys()) | set(freq2.keys())
    
    for label in all_labels:
        p1 = freq1.get(label, 0) / n
        p2 = freq2.get(label, 0) / n
        p_e += p1 * p2
    
    # Kappa
    if p_e == 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1 - p_e)
    
    return kappa