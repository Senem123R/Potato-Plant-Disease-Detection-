import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

# Load the model (modify the filename if needed)
MODEL = tf.keras.models.load_model(r"C:\Users\Risni Maleesha\Downloads\exam\exam\model_y.keras")



CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI app!"}


def read_file_as_image(data) -> np.ndarray:
      image = np.array(Image.open(BytesIO(data)))
      return image



@app.post("/predict")
async def predict(
      file: UploadFile = File(...)
):
      image = read_file_as_image(await file.read())
      img_batch = np.expand_dims(image, 0)

      prediction = MODEL.predict(img_batch)
      predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
      confidence = np.max(prediction[0])
      
      return {
          "class": predicted_class,
          "confidence": float(confidence)
      }

    

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)