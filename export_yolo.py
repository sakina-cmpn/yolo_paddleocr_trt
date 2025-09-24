import torch

def export_yolo(model_path="yolov5s.pt", onnx_path="yolov5s.onnx"):
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
    dummy = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=12, input_names=["images"], output_names=["output"]
    )
    print(f"YOLO model exported to {onnx_path}")

if __name__ == "__main__":
    export_yolo()
