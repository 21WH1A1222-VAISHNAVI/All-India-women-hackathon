# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'ALL.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Name = str(request.form['Name'])
        age = int(request.form['Age'])
        Gender = str(request.form['Gender'])
        Glucose = float(request.form['Glucose(mg/dL)'])
        Amylase = float(request.form['Amylase(U/L)'])
        cyto =  float(request.form['Cytokine level'])
        insu= float(request.form['Insulin(U/mL)'])
        ph = float(request.form['Salivary pH'])
        bmi = float(request.form['BMI'])
        salivasample = float(request.form['Saliva Sample(ml) '])
        bp = float(request.form['bloodpressure'])
        pep = float(request.form['C-Peptide level'])
        flow = float(request.form['Saliva Flowrate(ml/min)'])
        lipids = float(request.form['Lipids(mg/dL)'])
        immuno = float(request.form['Immunoglobulins(mg.dL)'])
        hb = float(request.form['HBA1c'])

        
        data = np.array([[Name,age,Gender,Glucose,Amylase,cyto,insu,ph,bmi,salivasample,bp,pep,flow,lipids,immuno,hb]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)