
import numpy as np

def plan_niveles(n=10, size_min=21, size_max=81, seed_base=7):
    sizes = np.linspace(size_min, size_max, n).astype(int)
    sizes = [s if s % 2 == 1 else s+1 for s in sizes]
    seeds = [seed_base + i*13 for i in range(n)]
    return list(zip(range(1,n+1), sizes, seeds))
