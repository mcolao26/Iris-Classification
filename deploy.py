from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# Load the model
model = pickle.load(open('savedmodel.sav', 'rb'))
@app.route('/')
def home():
    return render_template('index.html', **locals())

@app.route('/predict',methods=['POST','GET'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width  = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width  = float(request.form['petal_width'])
        
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    except (ValueError, KeyError):
        result = "⚠️ All text boxes must be filled with a number."
    
    return render_template('index.html', **locals())




if __name__ == '__main__':
    app.run(debug=True)
