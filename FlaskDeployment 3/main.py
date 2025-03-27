import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from util import base64_to_pil
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
import numpy as np
import google.generativeai as genai
import re


app = Flask(__name__)

# Register KerasLayer as a custom object
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the ML model once
model = tf.keras.models.load_model('models/efficientnet.h5', custom_objects=custom_objects)

# Configure Gemini API
genai.configure(api_key="AIzaSyC_iEbzi-vaFOh-V7ycI--kwvCNPsjy2fo")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model

    img_data = request.json.get("image", "")
    symptoms = request.json.get("symptoms", "")
    
    if not img_data and not symptoms:
        return jsonify({"error": "No image or symptoms data received"}), 400

    try:
        result_disease = "Unknown"
        if img_data:
            img = base64_to_pil(img_data)
            img = image.img_to_array(img) / 255.0
            img = tf.image.resize(tf.expand_dims(img, 0), (224, 224))

            y_pred = model.predict(img)
            val = np.argmax(y_pred)

            mapp = {1: 'Cataract', 0: 'Normal', 2: 'Diabetic Retinopathy', 3: 'Glaucoma'}
            result_disease = mapp.get(val, 'Unknown')

            print(result_disease,"result disease....")
            # result_disease = "Normal"
        if result_disease == "Normal" and not symptoms:
            return jsonify({
                "disease": "Normal",
                "general_tips": "Eat vitamin-rich foods, avoid screen strain, stay hydrated, and sleep well."
            })
        
        prompt = f"""
        Given the eye disease {result_disease}, along with the following symptoms provided by the user: "{symptoms}",
        provide a detailed recommendation. Include:
        - Lifestyle recommendations
        - Dietary recommendations
        - Adaptive screen time limits
        - Additional care guidelines specific to both the detected disease and mentioned symptoms.
        """
        
        response = gemini_model.generate_content(prompt)
        recommendation = response.text.strip() if response and hasattr(response, 'text') and response.text else "No response from AI."
        recommendation = re.sub(r"\*\*(.*?)\*\*", r"\1", recommendation)

        return jsonify({
            "disease": result_disease,
            "recommendation": recommendation
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "An error occurred while processing the request"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, threaded=False)
