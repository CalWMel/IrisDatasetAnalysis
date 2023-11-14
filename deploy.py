from flask import Flask, render_template, request
import pickle

# Create a Flask web application
app = Flask(__name__)

# Load the machine learning model from a saved file
model = pickle.load(open('savedmodel.sav', 'rb'))

# Define a route for the home page
@app.route('/')
def home():
    # Initialize the result variable
    result = ''
    # Render the 'index.html' template, passing the local variables
    return render_template('index.html', **locals())

# Define a route for prediction, which accepts both POST and GET requests
@app.route('/predict', methods=['POST', 'GET'])
def predict():

    # Extract input values from the submitted form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Use the loaded model to make a prediction based on the input values
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    
    # Render the 'index.html' template, passing the local variables
    return render_template('index.html', **locals())

# Start the Flask application if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)