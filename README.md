🥔 Potato Plant Disease Detection System

An End-to-End Deep Learning project in the agriculture domain that detects potato plant diseases using image classification with Convolutional Neural Networks (CNN).

Farmers often face economic loss due to plant diseases. This application allows users to upload or capture an image of a potato leaf, and the model predicts whether the plant is:

✅ Healthy

🍂 Early Blight

🌧 Late Blight

🚀 Project Overview

This project includes:

- CNN model trained using TensorFlow

- FastAPI backend for serving predictions

- Streamlit frontend for testing

- Docker support for deployment

Model optimization support (TensorFlow Lite & Quantization ready)


- 🛠️ Technology Stack


🤖 Model Building

- TensorFlow

- Convolutional Neural Networks (CNN)

- Data Augmentation

- tf.data Dataset API


⚙️ Backend & MLOps

- FastAPI

- Uvicorn

- TensorFlow Serving (optional)

- Docker


📉 Model Optimization

- Quantization

- TensorFlow Lite


🎨 Frontend

- Streamlit (Web UI)

- React JS (Planned)

- React Native (Planned Mobile App)


📂 Project Structure
Potato-Disease-Detection/
│
├── app.py                # FastAPI backend
├── main.py               # Streamlit frontend
├── model_y.keras         # Trained CNN model
├── fixed1_model.keras    # Streamlit model
├── requirements.txt
├── Dockerfile
├── project_pt.ipynb      # Model training notebook
└── README.md


🧠 Model Details

- Input Size: 256x256 images

- Architecture: CNN

- Output Classes:

- Early Blight,

- Late Blight,

- Healthy,

- Framework: TensorFlow / Keras


The model predicts:

- Disease class

- Confidence score

🌐 FastAPI Backend

Run Locally
pip install -r requirements.txt
uvicorn app:app --reload

API will run on:

http://localhost:8000

API Endpoints : 
GET /

Returns welcome message.

POST /predict

Upload an image file and get prediction.


Example response:

{
  "class": "Early Blight",
  "confidence": 0.98
}


🎨 Streamlit Frontend

Run Streamlit app:

streamlit run main.py

Features:

- Upload image

- Display image

- Predict disease

- Show classification result


🐳 Docker Deployment

Build Docker image:

docker build -t potato-disease-app .

Run container:

docker run -p 8000:8000 potato-disease-app


🔮 Future Improvements

- Deploy to AWS / Render / GCP

- Connect FastAPI with React frontend

- Build full mobile app using React Native

- Add more crop disease detection models

- Add model accuracy & confusion matrix visualization

- Implement CI/CD pipeline


📊 Use Case Impact

This system helps:

- Farmers detect diseases early

- Reduce crop loss

- Improve agricultural productivity

- Enable AI-powered smart farming
