from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_age=data["le_age"]
le_country = data["le_country"]
le_education = data["le_education"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        country = request.form['country']
        education = request.form['education']
        experience = float(request.form['experience'])

        X = np.array([[age,country, education, experience]])
        X[:, 0] = le_age.transform(X[:, 0])
        X[:, 1] = le_country.transform(X[:,1])
        X[:, 2] = le_education.transform(X[:,2])
        X = X.astype(float)

        salary = regressor.predict(X)
        return render_template('result.html', salary=salary[0])

if __name__ == '__main__':
    app.run(debug=True)
