import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO
import urllib.request

# Initialize the ONNX Runtime session
model_path = 'hair_classifier_empty.onnx'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

def preprocess_image(url):
    # 1. Download image
    with urllib.request.urlopen(url) as url_response:
        img_data = url_response.read()
    
    img = Image.open(BytesIO(img_data))
    
    # 2. Resize (CHECK HW8: usually 200x200 or 150x150)
    target_size = (200, 200) 
    img = img.resize(target_size, Image.NEAREST)
    
    # 3. Preprocessing (Rescale to 0-1 range)
    x = np.array(img, dtype=np.float32)
    x = x / 255.0
    
    # 4. Transpose to (Batch, Channel, Height, Width) if model expects it
    # Most PyTorch/ONNX models expect (N, C, H, W). 
    # Current shape is (H, W, C). We need (C, H, W).
    x = np.transpose(x, (2, 0, 1))
    
    # 5. Add Batch Dimension
    x = np.expand_dims(x, 0)
    
    return x

def predict(url):
    X = preprocess_image(url)
    
    # Run Inference
    outputs = session.run(None, {input_name: X})
    output = outputs[0]
    
    # Return the float value
    return float(output[0][0])

# Lambda Handler function
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return {'prediction': result}