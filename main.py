from picture import Pictures
from fastapi import FastAPI, Response, File, UploadFile
from matrix import Matrix
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
import base64
from PIL import Image
import io
import numpy as np

from unsupervised import SVD
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# init:
app = FastAPI()

# ==================== PUNTO 2 =============================

@app.get("/images-and-string", response_class=HTMLResponse)
async def get_images_and_string():
    # Load the images as PIL Image objects
    folder_path = "pictureFace"
    image_name = "ManuelaPiedrahita.jpeg"
    url = "https://drive.google.com/drive/folders/1f1aZ4i1lYsRaW9ID76iHfGztKdmAsg21?usp=sharing"
    pic = Pictures(folder_path, image_name, url)
    msj = pic.download_files()
    pic.load_pictures()
    pic.mean_picture()
    dist =  pic.distance()
    self_picture_path = pic.self_picture_path 
    mean_picture_path = pic.mean_path
    # read images:
    img1 = Image.open(self_picture_path)
    img2 = Image.open(mean_picture_path)
    # Convert the images to base64-encoded strings
    img1_buffer = io.BytesIO()
    img2_buffer = io.BytesIO()
    img1.save(img1_buffer, format='JPEG')
    img2.save(img2_buffer, format='JPEG')
    img1_base64 = base64.b64encode(img1_buffer.getvalue()).decode('utf-8')
    img2_base64 = base64.b64encode(img2_buffer.getvalue()).decode('utf-8')
    # Create the HTML string with the images and the message
    html_content = f"<h1> distancia = {str(dist)}</h1>"
    html_content += f"<img src='data:image/jpeg;base64,{img1_base64}' />"
    html_content += f"<img src='data:image/jpeg;base64,{img2_base64}' />"
    # Return the HTML response
    return HTMLResponse(content=html_content)

# ==================== PUNTO 11 =============================

@app.post("/image")
async def get_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')  # convert to grayscale
    img_array = np.array(img) # 2d
    image_1d = img_array.reshape(-1) # " 2d -> 1d"
    X = np.array([image_1d]) # sklearn
    # ======== dim red X input
    clf, svd = best_trained_model()
    X = svd.transform(X)
    print("================================")
    print(X.shape)
    y = clf.predict(X)
    print("================================")
    print(y)
    return {"class_predicted": y.tolist()}

# funcion auxiliar:

def best_trained_model():
    def load_mnist(digits_to_keep, N):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        arrays = []
        labels = []
        n = N // len(digits_to_keep)
        for cl in digits_to_keep:
            ind = y_train == cl
            A = x_train[ind]
            ycl = y_train[ind]
            arrays.append(A[:n])
            labels.append(ycl[:n])
        X = np.concatenate(tuple(arrays), axis = 0)
        y = np.concatenate(tuple(labels), axis = 0)
        a,b,c = X.shape
        X = X.reshape(a, b*c)
        return X, y
    # load mnist only for classes 0 and 8
    # load mnist only for classes 0 and 8
    train_images, train_labels = load_mnist([0,8], 200)
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.3, random_state=256)
    # ===================================
    unsupervised_svd = SVD(n_components=2)
    unsupervised_svd.fit(X_train)
    X_train_unsupervised_svd = unsupervised_svd.transform(X_train)
    X_test_unsupervised_svd = unsupervised_svd.transform(X_test)
    log_reg_with_unsupervised_svd = LogisticRegression(random_state=42)
    log_reg_with_unsupervised_svd.fit(X_train_unsupervised_svd, y_train)
    return log_reg_with_unsupervised_svd , unsupervised_svd
