balanced_pn_iter_bench: 2-bit (P,N) balanced add with carry, iterative loop
Iterative balanced PN add: N=1024, iters=128,  3187.96 µs/word,     2.57 Mtrits/s
Iterative balanced PN add: N=1024, iters=512, 13531.47 µs/word,     2.42 Mtrits/s
[NB] Iterative balanced PN add: N=1024, iters=128,     1.47 µs/word,  5577.39 Mtrits/s
[NB] Iterative balanced PN add: N=1024, iters=512,     5.84 µs/word,  5608.63 Mtrits/s
dashitest.old.keepme: harness-backed C_XOR benchmark
Correctness OK.
N=100000:  1359.57 µs/call     73.55 Mwords/s
dashitest.py: consumer benchmark
Implementation: C_XOR_array_swar from swar_test_harness (UFT-C semantics, specials quieted, per-word flags)
Correctness smoke: OK (matched harness reference on 10k words with specials)

Benchmarking naïve baseline (C_XOR_naive), full semantics.
Precomputed (stored) timings; run manually if you need to refresh:
N=     1000: 88474.91 µs/call       0.01 Mwords/s  (stored)
N=   100000: 8802841.42 µs/call       0.01 Mwords/s  (stored)

Benchmarking bitplane baseline (C_XOR_bitplane), normal lanes only (p_special=0).
N=     1000:   273.27 µs/call       3.66 Mwords/s
N=   100000: 31043.63 µs/call       3.22 Mwords/s

Benchmarking harness kernel (C_XOR_array_swar), specials enabled; no specials in inputs for throughput.
N=     1000:    29.35 µs/call      34.07 Mwords/s
N=   100000:   752.25 µs/call     132.93 Mwords/s
N=  5000000: 31806.73 µs/call     157.20 Mwords/s

Benchmarking dot product: reference vs SWAR (normal lanes only, p_special=0).
N=     1000: ref   101.72 µs/call   SWAR    16.65 µs/call   speedup x  6.1
N=   100000: ref 11399.21 µs/call   SWAR   528.86 µs/call   speedup x 21.6

Benchmarking threshold > 10: reference vs SWAR (normal lanes only, p_special=0).
N=     1000: ref   122.50 µs/call   SWAR    14.37 µs/call   speedup x  8.5
N=   100000: ref  7746.02 µs/call   SWAR   124.73 µs/call   speedup x 62.1
fused_iter_bench: XOR -> threshold -> dot loop (cache-resident)
Fused iter bench: N=1024, iters=256,    46.56 µs/iter,   791.73 Mop/s
Fused iter bench: N=1024, iters=1024,    53.95 µs/iter,   683.32 Mop/s
Sparse iterative classifier loop (XOR -> threshold -> dot) on cache-resident data.
K= 128: baseline    75.79 ms/epoch   SWAR    12.20 ms/epoch   speedup x 6.21
K= 512: baseline   321.29 ms/epoch   SWAR    77.48 ms/epoch   speedup x 4.15
Compiling Numba kernels (first call)...
OK: no_specials_small (N=10000, p_special=0.0)
OK: rare_specials (N=200000, p_special=0.0001)
OK: some_specials (N=200000, p_special=0.01)
OK: many_specials (N=50000, p_special=0.2)
BENCH (SWAR candidate): N=     1000     19.26 µs/call     51.93 Mwords/s
BENCH (SWAR candidate): N=   100000   1025.50 µs/call     97.51 Mwords/s
BENCH (SWAR candidate): N=  5000000  27135.95 µs/call    184.26 Mwords/s
All tests passed.
Triadic NN bench: baseline (unpacked int8) vs SWAR packed dot_product
Neurons: 8, input lanes: 12, values in {0,1,2}
N=     1000: baseline    74.55 µs/call ( 1287.71 Mop/s)  SWAR   252.16 µs/call (  380.71 Mop/s)  speedup x  0.3
N=   100000: baseline  5623.10 µs/call ( 1707.24 Mop/s)  SWAR  7810.94 µs/call ( 1229.05 Mop/s)  speedup x  0.7
triadic_nn_bench2: baseline NumPy vs packed SWAR dot_product_swar
N=     1000, M=  8: baseline    79.37 µs/call ( 1209.54 Mop/s) SWAR   202.35 µs/call (  474.42 Mop/s) speedup x 0.39
N=   100000, M= 16: baseline 13097.40 µs/call ( 1465.94 Mop/s) SWAR 17307.89 µs/call ( 1109.32 Mop/s) speedup x 0.76
