from flask import Flask,render_template_string,request
import torch
import torch.nn as nn
import joblib

model = torch.jit.load('irrigation.pt')
model.eval()

randomforest = joblib.load("irrigation.rn")

template = """
<html>
<body>
        <form method="POST">
<label for="decimal1">CropType: Select the type of crop you are growing</label>
<select id="decimal1" name="decimal1" required>
    <option value="" disabled selected>Select a crop</option>
    <option value="Wheat">Wheat</option>
    <option value="Groundnuts">Groundnuts</option>
    <option value="Garden Flowers">Garden Flowers</option>
    <option value="Maize">Maize</option>
    <option value="Paddy">Paddy</option>
    <option value="Potato">Potato</option>
    <option value="Pulse">Pulse</option>
    <option value="Sugarcane">Sugarcane</option>
    <option value="Coffee">Coffee</option>
</select>
<p>Please select one of the available crops from the dropdown.</p>
<br><br>

<label for="decimal2">CropDays: Enter the number of days the crop has been growing</label>
<input type="number" id="decimal2" name="decimal2" required>
<p>Enter the duration (in days) since the crop was planted.</p>
<br><br>

<label for="decimal3">SoilMoisture: Enter the current moisture level in the soil</label>
<input type="number" id="decimal3" name="decimal3" required>
<p>Provide the soil moisture level as a percentage (e.g., 30 for 30%).</p>
<br><br>

<label for="decimal4">Temperature: Enter the current temperature in degrees Celsius</label>
<input type="number" id="decimal4" name="decimal4" required>
<p>Input the average temperature in Celsius during the crop's growth period.</p>
<br><br>

<label for="decimal5">Humidity: Enter the current humidity level</label>
<input type="number" id="decimal5" name="decimal5" required>
<p>Provide the humidity level as a percentage (e.g., 70 for 70%).</p>
<br><br>

        <button type="submit">Submit</button>
    </form>
    {% if randout %}
    <h2>You should {{final}} </h2>
    {% else %}
        <h2>Please fill the detail to know about irrigation</h2>
    {% endif %}
</body>
</html>
"""
#for crop_recommendation
encode = {"Wheat" : 0,
    "Groundnuts" : 1,
    "Garden Flowers" : 2,
    "Maize" : 3,
    "Paddy" : 4,
    "Potato" : 5,
    "Pulse" : 6,
    "Sugarcane" : 7,
    "Coffee" : 8
}
encode_pair = list(encode.items())

def encr(val):
    if(val == 0):
        return "Irrigate"
    else: 
        return "Not Irrigate"

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':

        data = [
            request.form['decimal1'],
            request.form['decimal2'],
            request.form['decimal3'],
            request.form['decimal4'],
            request.form['decimal5'],
        ]
        
        m = list()
        for j in data:
            if(j.isalpha() or j == "Garden Flowers"):
                m.append(encode[j])
            else:
                m.append(float(j))
        print(m)
        inp = torch.tensor([m[0],m[1],m[2],m[3],m[4]], dtype= torch.float32)
        out = model.forward(inp)
        _, out = torch.max(out, axis = 0)
        out = out.item();
        outf = encr(out+1)
        
        out2 = randomforest.predict(inp.view(1,5).numpy())
        out2f = encr(out2[0])
        out3 = ((out+1)*0.2 + out2[0]*0.8)//1
        out3f = encr(out3)
        return render_template_string(template, data=outf,randout = out2f,final = out3f)
    return render_template_string(template)


if __name__ == '__main__':

    app.run(debug=True)
