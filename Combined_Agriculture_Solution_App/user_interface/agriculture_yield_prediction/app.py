from flask import Flask,render_template_string,request
import torch
import torch.nn as nn
import joblib

#creating the model for prediction part
class Prediction_Model(nn.Module):
    def __init__(self):
        super(Prediction_Model,self).__init__()
        self.l1 = nn.Linear(9,100)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(100,50)
        self.l3 = nn.Linear(50,1)
    def forward_training(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = torch.jit.load('agriculture_yield_prediction.pt')
model.eval()

randomforest = joblib.load("agricultural_yield_prediction.rn")

template = """
<html>
<body>
        <form method="POST">
        <label for="input1">Region: The geographical region where the crop is grown (North, East, South, West): </label>
<select id="input1" name="input1" required>
    <option value="" disabled selected>Select a region</option>
    <option value="North">North</option>
    <option value="East">East</option>
    <option value="South">South</option>
    <option value="West">West</option>
</select>
<br><br>

<label for="input2">Soil_Type: The type of soil in which the crop is planted (Clay, Sandy, Loam, Silt, Peaty, Chalky): </label>
<select id="input2" name="input2" required>
    <option value="" disabled selected>Select a soil type</option>
    <option value="Clay">Clay</option>
    <option value="Sandy">Sandy</option>
    <option value="Loam">Loam</option>
    <option value="Silt">Silt</option>
    <option value="Peaty">Peaty</option>
    <option value="Chalky">Chalky</option>
</select>
<br><br>

<label for="input3">Crop: The type of crop grown (Wheat, Rice, Maize, Barley, Soybean, Cotton): </label>
<select id="input3" name="input3" required>
    <option value="" disabled selected>Select a crop</option>
    <option value="Wheat">Wheat</option>
    <option value="Rice">Rice</option>
    <option value="Maize">Maize</option>
    <option value="Barley">Barley</option>
    <option value="Soybean">Soybean</option>
    <option value="Cotton">Cotton</option>
</select>
<br><br>

<label for="input4">Fertilizer_Used: Indicates whether fertilizer was applied (True = Yes, False = No): </label>
<select id="input4" name="input4" required>
    <option value="" disabled selected>Select an option</option>
    <option value="True">True</option>
    <option value="False">False</option>
</select>
<br><br>

<label for="input5">Irrigation_Used: Indicates whether irrigation was used during the crop growth period (True = Yes, False = No): </label>
<select id="input5" name="input5" required>
    <option value="" disabled selected>Select an option</option>
    <option value="True">True</option>
    <option value="False">False</option>
</select>
<br><br>

<label for="input6">Weather_Condition: The predominant weather condition during the growing season (Sunny, Rainy, Cloudy): </label>
<select id="input6" name="input6" required>
    <option value="" disabled selected>Select a weather condition</option>
    <option value="Sunny">Sunny</option>
    <option value="Rainy">Rainy</option>
    <option value="Cloudy">Cloudy</option>
</select>
<br><br>

        <label for="decimal1">Rainfall_mm: The amount of rainfall received in millimeters during the crop growth period :</label>
        <input type="number" id="decimal1" name="decimal1" required>
        <br><br>

        <label for="decimal2">Temperature_Celsius: The average temperature during the crop growth period, measured in degrees Celsius :</label>
        <input type="number" id="decimal2" name="decimal2" required>
        <br><br>

        <label for="decimal3">Days_to_Harvest: The number of days taken for the crop to be harvested after planting :</label>
        <input type="number" id="decimal3" name="decimal3" required>
        <br><br>

        <button type="submit">Submit</button>
    </form>
    {% if data %}
        <h2>Predicted Yield</h2>
        <h3>{{final}} tonnes per hectare</h3>
    {% else %}
        <h2>Please fill the detail to know the Predicted Yield</h2>
    {% endif %}
</body>
</html>
"""
encode = {
"North" : 0,
"East" : 1,
"South" : 2,
"West" : 3,
"Clay" : 0,
"Sandy" : 1,
"Loam" : 2,
"Silt" : 3,
"Peaty" : 4,
"Chalky" : 5,
"Wheat" : 0,
"Rice" : 1,
"Maize" : 2,
"Barley" : 3,
"Soybean" : 4,
"Cotton" : 5,
"True" : 1,
"False" : 0,
"Sunny" : 0,
"Rainy" : 1,
"Cloudy" : 2,
}

encode_pair = list(encode.items())

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':

        data = [
            request.form['input1'],
            request.form['input2'],
            request.form['input3'],
            request.form['input4'],
            request.form['input5'],
            request.form['input6'],
            request.form['decimal1'],
            request.form['decimal2'],
            request.form['decimal3']
        ]
        
        m = list()
        for j in data:
            if(j.isalpha()):
                m.append(encode[j])
            else:
                m.append(float(j))
        print(m)
        inp = torch.tensor([m[0],m[1],m[2],m[6],m[7],m[3],m[4],m[5],m[8]], dtype= torch.float32)
        out = model.forward(inp)
        out2 = randomforest.predict(inp.view(1,9).numpy())
        out3 = out*0.2 + out2[0]*0.8
        return render_template_string(template, data=out.item(),randout = out2[0],final = out3.item())
    return render_template_string(template)


if __name__ == '__main__':

    app.run(debug=True)
