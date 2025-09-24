import tensorrt as trt
import numpy as np

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data):
        super().__init__()
        self.data = calibration_data
        self.index = 0

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.index >= len(self.data):
            return None
        batch = self.data[self.index]
        self.index += 1
        return [batch]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        with open("calib_cache.bin", "wb") as f:
            f.write(cache)

if __name__ == "__main__":
    dummy_data = [np.random.rand(1, 3, 640, 640).astype(np.float32) for _ in range(5)]
    calib = EntropyCalibrator(dummy_data)
    print("Calibration ready.")
