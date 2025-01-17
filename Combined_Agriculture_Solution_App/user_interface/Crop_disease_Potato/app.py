import os
from flask import Flask, request, render_template_string, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms


app = Flask(__name__)

# Load the trained PyTorch model


model = torch.jit.load('crop_disease_potato.pt')
model.eval()
# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the input size of your CNN
    transforms.ToTensor(),         # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

template = '''
    <!doctype html>
    <title>Image Classification</title>
    <h1>Upload an image</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload and Classify">
    </form>

    <h1> This might be {{data}}
    '''
@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Image Classification</title>
    <h1>Upload an image</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload and Classify">
    </form>
    '''

def pred(i):
    ecode = {0 : 'Potato___Early_Blight' ,
1 : 'Potato___Healthy' ,
2 : 'Potato___Late_Blight'}
    return ecode[i]


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and preprocess the image
    image = Image.open(file_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict using the model
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    # Cleanup uploaded file
    os.remove(file_path)

    # Return prediction result
    class_idx = predicted.item()
    out = pred(class_idx)
    return render_template_string(template, data=out)

if __name__ == '__main__':
    app.run(debug=True)
