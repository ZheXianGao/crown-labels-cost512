import numpy as np

def validate(labels, families, parents):
    active = set(np.unique(labels[labels > 0]))
    covered = set()
    for p in active & set(parents):
        members = set(parents[p])
        if members & active or members & covered:
            raise ValueError('Overlapping parent/child explanations')
        covered.update(members)
