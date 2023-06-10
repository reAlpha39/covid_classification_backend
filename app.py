import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI()

# model ML
model_ctscan_vgg = None
model_ctscan_resnet = None
model_xray_vgg = None
model_xray_resnet = None

# load model ML
def load_model():
    global model_ctscan_vgg, model_ctscan_resnet, model_xray_vgg, model_xray_resnet
    model_ctscan_vgg = load_model('models/ctscan_vgg.h5')
    model_ctscan_resnet = load_model('models/ctscan_resnet.h5')
    model_xray_vgg = load_model('models/xray_vgg.h5')
    model_xray_resnet = load_model('models/xray_resnet.h5')

# predict image vgg
def predict_image_vgg(image, model):
    image = image.load_img(image, target_size=(512, 512))
    image_array = image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.

    prediction = model.predict(image_array)
    class_names = ['covid', 'noncovid']
    predicted_class = np.argmax(prediction[0])
    class_label = class_names[predicted_class]
    confidence = prediction[0][predicted_class]


    print("Predicted class:", class_label)
    print("Confidence:", confidence)

    return prediction[0], class_label, confidence


def predict_image_resnet(image, model):
    image = image.load_img(image, target_size=(512, 512))
    image_array = image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.

    prediction = model.predict(image_array)
    result = ''
    if prediction[0] >= 0.5:
        result = "COVID"
    else:
        result = "non-COVID-19"
    return prediction[0], result



@app.get("/api")
async def root():
    return {"message": "Hello World"}

@app.post("/api/ctscan/vgg")
async def predict_ctscan_vgg(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.seek(0)

    raw, class_label, confidence = predict_ctscan_vgg(temp_file.name, model_ctscan_vgg)

    os.remove(temp_file.name)

    return {
        "status": "success",
        "data": {
            "raw": raw,
            "class_labe": class_label,
            "confidence": confidence
        }
    }

@app.post("/api/ctscan/resnet")
async def predict_ctscan_resnet(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.seek(0)

    raw, result = predict_image_resnet(temp_file.name, model_ctscan_resnet)

    os.remove(temp_file.name)

    return {
        "status": "success",
        "data": {
            "raw": raw,
            "result": result
        }
    }

@app.post("/api/xray/vgg")
async def predict_xray_vgg(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.seek(0)

    raw, class_label, confidence = predict_image_vgg(temp_file.name, model_xray_vgg)

    os.remove(temp_file.name)

    return {
        "status": "success",
        "data": {
            "raw": raw,
            "class_labe": class_label,
            "confidence": confidence
        }
    }

@app.post("/api/xray/resnet")
async def predict_xray_resnet(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file.seek(0)

    raw, result = predict_image_resnet(temp_file.name, model_xray_resnet)

    os.remove(temp_file.name)

    return {
        "status": "success",
        "data": {
            "raw": raw,
            "result": result
        }
    }

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=5555)
