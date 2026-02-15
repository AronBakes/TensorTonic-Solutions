import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    if len(rater1) != len(rater2):
          raise ValueError("Both lists must have the same length")
      
    n = len(rater1)
    
    # Build frequency maps
    freq1 = {}
    freq2 = {}

    # Observed agreement (p_o)
    agreements = 0
    for i in range(n):
        label1 = rater1[i]
        label2 = rater2[i]

        freq1[label1] = freq1.get(label1, 0) + 1
        freq2[label2] = freq2.get(label2, 0) + 1
      
        if label1 == label2:
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