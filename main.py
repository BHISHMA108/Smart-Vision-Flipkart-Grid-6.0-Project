import cv2
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

# Load the YOLO model
weights_path = 'Yolo/yolov3.weights'  # Update with your YOLO weights path if necessary
config_path = 'Yolo/yolov3.cfg'        # Update with your YOLO config path if necessary
labels_path = 'Yolo/coco.names'       # Use your custom classes file (mango.names)

net = cv2.dnn.readNet(weights_path, config_path)

# Load the MobileNet model
mobilenet_model = load_model('converted_keras/keras_model.h5')  # Updated path

# Load custom class labels for mango
with open(labels_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Function to predict ripeness stage
def predict_mango_ripeness(image_array):
    img_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = mobilenet_model.predict(img_array)
    class_indices = {0: 'FullyRipe', 1: 'Overripe', 2: 'Perished', 3: 'Semiripe', 4: 'Unripe'}

    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_indices[predicted_class]

# Function to detect mangoes in an image
def detect_and_classify_mangoes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Run forward pass
    detections = net.forward(output_layers)

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check for mango detection (adjust class_id according to your coco.names)
            if class_id == 0 and confidence > 0.5:  # Adjust confidence threshold as necessary
                # Get the bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the mango region
                mango_crop = image[y:y + h, x:x + w]
                # Resize for MobileNet input
                mango_crop_resized = cv2.resize(mango_crop, (224, 224))

                # Predict ripeness
                ripeness_stage = predict_mango_ripeness(mango_crop_resized)

                # Put label on the image
                cv2.putText(image, ripeness_stage, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the result
    cv2.imshow('Mango Detection and Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'Test_Img_1.jpg'  # Replace with your image path
detect_and_classify_mangoes(image_path)
