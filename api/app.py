import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = FastAPI()

# model ML
model_ctscan_vgg = None
model_ctscan_resnet = None
model_xray_vgg = None
model_xray_resnet = None

# load model ML
def load_model_ml():
    global model_ctscan_vgg, model_ctscan_resnet, model_xray_vgg, model_xray_resnet
    model_ctscan_vgg = load_model('./models/ct_vgg.h5')
    model_ctscan_resnet = load_model('./models/ct_resnet.h5')
    model_xray_vgg = load_model('./models/xray_vgg.h5')
    model_xray_resnet = load_model('./models/xray_resnet.h5')

# predict image vgg
def predict_image(image_input, model):
    img = image.load_img(image_input, target_size=(224, 224))
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.

    prediction = model.predict(image_array)
    if prediction[1] > 0.5:
        class_label = "COVID-19"
        confidence = prediction[1]*100
    else:
        class_label = "non-COVID-19"
        confidence = (1 - prediction[1])*100


    print("Predicted class:", class_label)
    print("Confidence:", confidence)

    return prediction, class_label, confidence



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ctscan/vgg")
async def predict_ctscan_vgg(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.seek(0)

        raw, class_label, confidence = predict_image(temp_file.name, model_ctscan_vgg)

        os.remove(temp_file.name)

        return {
            "status": "success",
            "data": {
                "raw": raw,
                "class_labe": class_label,
                "confidence": confidence
            }
        }

@app.post("/ctscan/resnet")
async def predict_ctscan_resnet(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.seek(0)

        raw, class_label, confidence = predict_image(temp_file.name, model_xray_vgg)

        os.remove(temp_file.name)

        return {
            "status": "success",
            "data": {
                "raw": raw,
                "class_labe": class_label,
                "confidence": confidence
            }
        }

@app.post("/xray/vgg")
async def predict_xray_vgg(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.seek(0)

        raw, class_label, confidence = predict_image(temp_file.name, model_xray_vgg)

        os.remove(temp_file.name)

        return {
            "status": "success",
            "data": {
                "raw": raw,
                "class_labe": class_label,
                "confidence": confidence
            }
        }

@app.post("/xray/resnet")
async def predict_xray_resnet(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.seek(0)

        raw, class_label, confidence = predict_image(temp_file.name, model_xray_vgg)

        os.remove(temp_file.name)

        return {
            "status": "success",
            "data": {
                "raw": raw,
                "class_labe": class_label,
                "confidence": confidence
            }
        }

if __name__ == "__main__":
    load_model_ml()
    uvicorn.run(app, host="0.0.0.0", port=5555)
