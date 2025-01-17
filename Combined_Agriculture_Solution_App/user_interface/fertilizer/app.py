from flask import Flask,render_template_string,request
import torch
import torch.nn as nn
import joblib

model = torch.jit.load('fertilizer.pt')
model.eval()

randomforest = joblib.load("fertilizer.rn")

template = """
<html>
<body>
        <form method="POST">
<label for="decimal1">Temperature: Enter the current temperature in degrees Celsius</label>
<input type="number" id="decimal1" name="decimal1" required>
<p>Input the average temperature in Celsius during the crop's growth period.</p>
<br><br>

<label for="decimal2">Humidity: Enter the current humidity level</label>
<input type="number" id="decimal2" name="decimal2" required>
<p>Provide the humidity level as a percentage (e.g., 70 for 70%).</p>
<br><br>

<label for="decimal3">Moisture: Enter the current moisture level in the soil</label>
<input type="number" id="decimal3" name="decimal3" required>
<p>Provide the soil moisture level as a percentage (e.g., 30 for 30%).</p>
<br><br>

<label for="decimal4">Soil Type: Select the type of soil</label>
<select id="decimal4" name="decimal4" required>
    <option value="" disabled selected>Select soil type</option>
    <option value="Sandy">Sandy</option>
    <option value="Loamy">Loamy</option>
    <option value="Black">Black</option>
    <option value="Red">Red</option>
    <option value="Clayey">Clayey</option>
</select>
<p>Please select the soil type from the dropdown.</p>
<br><br>

<label for="decimal5">Crop Type: Select the type of crop you are growing</label>
<select id="decimal5" name="decimal5" required>
    <option value="" disabled selected>Select a crop</option>
    <option value="Maize">Maize</option>
    <option value="Sugarcane">Sugarcane</option>
    <option value="Cotton">Cotton</option>
    <option value="Tobacco">Tobacco</option>
    <option value="Paddy">Paddy</option>
</select>
<p>Please select the crop type from the dropdown.</p>
<br><br>

<label for="decimal6">Nitrogen: Enter the nitrogen level</label>
<input type="number" id="decimal6" name="decimal6" required>
<p>Input the nitrogen level in the soil.</p>
<br><br>

<label for="decimal7">Potassium: Enter the potassium level</label>
<input type="number" id="decimal7" name="decimal7" required>
<p>Input the potassium level in the soil.</p>
<br><br>

<label for="decimal8">Phosphorous: Enter the phosphorous level</label>
<input type="number" id="decimal8" name="decimal8" required>
<p>Input the phosphorous level in the soil.</p>
<br><br>

        <button type="submit">Submit</button>
    </form>
    {% if randout %}
        <h2>Suggested Fertilizer</h2>
        <h3> {{final}} </h3>
    {% else %}
        <h2>Please fill the detail to know the fertilizer</h2>
    {% endif %}
</body>
</html>
"""
encode = {'Sandy': 0, 'Maize': 0, 'Urea': 0, 'Loamy': 1, 'Sugarcane': 1, 'DAP': 1, 'Black': 2, 'Cotton': 2, '14-35-14': 2, 'Red': 3, 'Tobacco': 3, '28-28': 3, 'Clayey': 4, 'Paddy': 4, 'Barley': 5, '17-17-17': 4, '20-20': 5, 'Wheat': 6, 'Millets': 7, 'Oil seeds': 8, 'Pulses': 9, 'Ground Nuts': 10, '10-26-26': 6}

encode_pair = list(encode.items())

def encr(val):
    rev_encode = {val : key for key, val in encode.items()}
    return rev_encode[val]

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
            request.form['decimal6'],
            request.form['decimal7'],
            request.form['decimal8'],
        ]
        
        m = list()
        for j in data:
            if(j in encode.keys()):
                m.append(encode[j])
            else:
                m.append(float(j))
        print(m)
        inp = torch.tensor([m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7]], dtype= torch.float32)
        out = model.forward(inp)
        _, out = torch.max(out, axis = 0)
        out = out.item()
        outf = encr(out+1)
        
        out2 = randomforest.predict(inp.view(1,8).numpy())
        out2f = encr(out2[0])

        out3 = round(0.2 * out + 0.8 * out2[0])
        out3f = encr(out3)
        return render_template_string(template, data=outf,randout = out2f, final = out3f)
    return render_template_string(template)


if __name__ == '__main__':

    app.run(debug=True)
