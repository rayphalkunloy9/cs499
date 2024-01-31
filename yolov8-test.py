import matplotlib.pyplot as plt
from ultralytics import YOLO

# Define your custom model configuration
model = YOLO('./weight/best.pt')
# Load the trained weights
img = './image/Car449.jpg'  # Replace with the path to your image

# Run inference on 'bus.jpg' with arguments
model.predict(img, save=True, conf=0.5)