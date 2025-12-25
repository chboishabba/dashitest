import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from compression.compression_bench import run_benchmark


def test_compression_benchmark_smoke():
    results = run_benchmark(height=12, width=12, steps=10, seed=1)
    assert results, "benchmark returned no results"
    for res in results:
        assert res["symbols"] > 0
        assert "compressed" in res
        for codec in ("lzma", "gzip", "zlib"):
            comp = res["compressed"][codec]
            assert comp.bytes_out > 0
            assert comp.ms >= 0.0
