from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import time
import os
from joblib import load
from predict_logic import preprocess_img, sliding_window, pyramid, nms, visualize_bbox, predict

# Initialize Flask app
app = Flask(__name__)

# Load saved model and scaler
model = load(r"C:\GitHub\ML project\SVM-TrafficSigns\UI\svm_model.pkl")
scaler = load(r"C:\GitHub\ML project\SVM-TrafficSigns\UI\scaler.pkl")
label_encoder = load(r"C:\GitHub\ML project\SVM-TrafficSigns\UI\label_encoder.pkl")

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the 'templates' folder

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image file!"}), 400

        # Perform prediction
        annotated_img, bboxes = predict(img, label_encoder)

        # Save the annotated image
        output_path = r"C:\GitHub\ML project\SVM-TrafficSigns\UI\static\output_image.png"
        success = cv2.imwrite(output_path, annotated_img)
        print(f"Image saved successfully: {success}, Path: {output_path}")
        if not success:
            raise Exception("Failed to save the output image.")

        # Add a timestamp to the image URL to prevent caching
        timestamp = int(time.time())
        return jsonify({"image_url": f"/static/output_image.png?v={timestamp}"})


    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
