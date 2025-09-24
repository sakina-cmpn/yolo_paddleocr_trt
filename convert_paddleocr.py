from paddleocr import PaddleOCR

def export_paddleocr(onnx_path="paddleocr.onnx"):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print(f"Simulating PaddleOCR export... (would be saved to {onnx_path})")

if __name__ == "__main__":
    export_paddleocr()
