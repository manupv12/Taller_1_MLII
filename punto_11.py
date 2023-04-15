import uvicorn
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

@app.post("/image")
async def get_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')  # convert to grayscale
    img_array = np.array(img)
    return {"image_values": img_array.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)