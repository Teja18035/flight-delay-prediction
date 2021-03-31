import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model = load("lr.save")
ss = load("transform.save")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():

    dep_time = request.form['dep_time']
    dep_delay = request.form['dep_delay']
    arr_time = request.form['arr_time']
    arr_delay = request.form['arr_delay']
    
    total = [[dep_time,dep_delay,arr_time,arr_delay]]
    prediction = model.predict(ss.transform(total))
    
    if (prediction == 0):
        output = "No Delay. the flight reaches in correct time"
    else:
        output = "The flight may be delayed"
    
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)