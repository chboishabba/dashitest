import os, random, math, zlib
from collections import Counter

# Generate random data
N = 4096
data = os.urandom(N)

def entropy(seq):
    c = Counter(seq)
    total = len(seq)
    return -sum((v/total)*math.log2(v/total) for v in c.values())

# Raw byte entropy
raw_entropy = entropy(data)

# Convert bytes to base-3 digits (6 trits per byte)
trits = []
for b in data:
    x = b
    for _ in range(6):
        trits.append(x % 3)
        x //= 3

trit_entropy = entropy(trits)

# Compress both with zlib (proxy entropy coder)
raw_z = len(zlib.compress(data))
trits_bytes = bytes(trits)
trits_z = len(zlib.compress(trits_bytes))

(raw_entropy, trit_entropy, raw_z, trits_z)
