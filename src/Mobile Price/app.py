from flask import Flask, render_template, request
import pickle as pkl
import numpy as np

# Load the trained model (assuming it is saved as modelrfc.pkl)
model = pkl.load(open('model_lr_grid.pkl', 'rb'))

# Initialize the flask app
app = Flask(__name__)

# Default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # For rendering results on HTML GUI
    if request.method == "POST":
        # Extract form data
        battery_power = int(request.form['battery_power'])
        blue = int(request.form['blue'])
        clock_speed = float(request.form['clock_speed'])
        dual_sim = int(request.form['dual_sim'])
        fc = int(request.form['fc'])
        four_g = int(request.form['four_g'])
        int_memory = int(request.form['int_memory'])
        m_dep = float(request.form['m_dep'])
        mobile_wt = int(request.form['mobile_wt'])
        n_cores = int(request.form['n_cores'])
        pc = int(request.form['pc'])
        px_height = int(request.form['px_height'])
        px_width = int(request.form['px_width'])
        ram = int(request.form['ram'])
        sc_h = int(request.form['sc_h'])
        sc_w = int(request.form['sc_w'])
        talk_time = int(request.form['talk_time'])
        three_g = int(request.form['three_g'])
        touch_screen = int(request.form['touch_screen'])
        wifi = int(request.form['wifi'])

        # Prepare the input for the model
        features = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt,
                              n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Map prediction to price range
        price_range = ["Low", "Medium", "High", "Very High"]
        predicted_price_range = price_range[prediction]

        return render_template('output.html', prediction=predicted_price_range)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
