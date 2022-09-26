from io import BytesIO
import tensorflow as tf
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    'http://localhost',
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

MODEL = tf.keras.models.load_model("predict_ver(1.0.0)")

CLASSNAMES = ['Early Blight', "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am zinda"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(image_batch)
    index = np.argmax(prediction[0])
    confidence =np.max(prediction[0])
    print("Class: ",CLASSNAMES[index])
    print("Confidence: ", float(confidence))
    return {
        "class": CLASSNAMES[index],
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000) 
