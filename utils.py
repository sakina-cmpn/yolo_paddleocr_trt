import cv2

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    return img

def postprocess(output):
    print("Simulated postprocess on:", output)
