from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
with open('savedmodel.sav', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html', result='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Perform input validation if needed

        # Make a prediction using the model
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        return render_template('index.html', result=result)
    except ValueError:
        return render_template('index.html', result='Invalid input')

if __name__ == '__main__':
    app.run(debug=True)
