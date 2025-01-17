from flask import Flask,render_template_string,request
import torch
import torch.nn as nn
import joblib

model = torch.jit.load('crop_recommendation.pt')
model.eval()

randomforest = joblib.load("crop_recommendation.rn")

template = """
<html>
<body>
        <form method="POST">
        <label for="decimal1">Quantity of nitrogen in the soil</label>
        <input type="number" id="decimal1" name="decimal1" required>
        <br><br>

        <label for="decimal2">Quantity of phosphorous in the soil</label>
        <input type="number" id="decimal2" name="decimal2" required>
        <br><br>

        <label for="decimal3">Quantity of potassium in the soil</label>
        <input type="number" id="decimal3" name="decimal3" required>
        <br><br>
        
        <label for="decimal4"> Average Temperature</label>
        <input type="number" id="decimal4" name="decimal4" required>
        <br><br>

        <label for="decimal5">Average Rainfall </label>
        <input type="number" id="decimal5" name="decimal5" required>
        <br><br>

        <label for="decimal6">PH of the soil</label>
        <input type="number" id="decimal6" name="decimal6" required>
        <br><br>

        
        <label for="decimal7">Average Rainfall </label>
        <input type="number" id="decimal7" name="decimal7" required>
        <br><br>

        <button type="submit">Submit</button>
    </form>
    {% if randout %}
        <h2> The best crop to grow will be {{final}} </h2>
    {% else %}
        <h2>Please fill the detail to know the Predicted Crop</h2>
    {% endif %}
</body>
</html>
"""
#for crop_recommendation
encode = {
    "rice" : 1,
    "maize" : 2,
    "chickpea" : 6,
    "kidneybeans" : 7,
    "pigeonpeas" : 8,
    "mothbeans" : 9,
    "mungbean" : 10,
    "blackgram" : 11,
    "lentil" : 12,
    "pomegranate" : 13,
    "banana" : 14,
    "mango" : 15,
    "grapes" : 16,
    "watermelon" : 17,
    "muskmelon" : 18,
    "apple" : 19,
    "orange" : 20,
    "papaya" : 21,
    "coconut" : 22,
    "cotton" : 23,
    "jute" : 24,
    "coffee" : 25
}
encode_pair = list(encode.items())

def encr(val):
    decode = {value : key for key,value in encode_pair}
    if(val in decode.keys()):
        return decode[val]
    else:
        return "no suggestion"

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
        ]
        
        m = list()
        for j in data:
            if(j.isalpha()):
                m.append(encode[j])
            else:
                m.append(float(j))
        print(m)
        inp = torch.tensor([m[0],m[1],m[2],m[3],m[4],m[5],m[6]], dtype= torch.float32)
        out = model.forward(inp)
        _, out = torch.max(out, axis = 0)
        out = out.item();
        outf = encr(out+1)
        
        out2 = randomforest.predict(inp.view(1,7).numpy())
        out2f = encr(out2[0])

        out3 = (0.2*(out+1) + 0.8*(out2[0]))/1
        out3f = encr(out3)
        return render_template_string(template, data=outf,randout = out2f, final = out3f)
    return render_template_string(template)


if __name__ == '__main__':

    app.run(debug=True)
