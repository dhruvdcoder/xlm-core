import random
import numpy as np
import torch
import time


def avg_shuffle(n, reps=1000):
    orig = list(range(n))
    t0 = time.perf_counter()
    for _ in range(reps):
        arr = orig.copy()
        random.shuffle(arr)
    return (time.perf_counter() - t0) / reps


def avg_np(n, reps=1000):
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = np.random.permutation(n)
    return (time.perf_counter() - t0) / reps


def avg_torch(n, reps=1000):
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.randperm(n)
    return (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    n = 2000  # change to whatever you like
    reps = 1000  # number of trials for averaging

    t_shuffle = avg_shuffle(n, reps)
    t_np = avg_np(n, reps)
    t_torch = avg_torch(n, reps)

    print(f"n={n}, averaged over {reps} runs:")
    print(f"  random.shuffle:       {t_shuffle*1e6:.2f} μs per call")
    print(f"  np.random.permutation: {t_np*1e6:.2f} μs per call")
    print(f"  torch.randperm:        {t_torch*1e6:.2f} μs per call")

# Results:
# random.shuffle:       376.64 μs per call
# np.random.permutation: 25.46 μs per call
# torch.randperm:        44.70 μs per call
