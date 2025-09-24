# YOLO + PaddleOCR with TensorRT

## Steps
1. Export YOLO to ONNX  
   ```bash
   python export_yolo.py

## Steps
2. Export PaddleOCR (simulated)  
   ```bash
   python convert_paddleocr.py

## Steps
3. Build TensorRT engine 
   ```bash
   python trt_builder.py

## Steps
4. Run Benchmark
   ```bash
   python benchmark.py

