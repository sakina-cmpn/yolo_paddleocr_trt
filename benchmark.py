import time
import numpy as np

def benchmark_inference(dummy_runs=100):
    start = time.time()
    for _ in range(dummy_runs):
        _ = np.random.rand(1, 3, 640, 640)  # simulate inference
    end = time.time()
    print(f"Simulated {dummy_runs} inferences in {end - start:.2f}s")

if __name__ == "__main__":
    benchmark_inference()
