from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Gender=request.form.get('Gender'),
            Married=request.form.get('Married'),
            Dependents=request.form.get('Dependents'),
            Education=request.form.get('Education'),
            Self_Employed=request.form.get('Self_Employed'),
            ApplicantIncome=float(request.form.get('ApplicantIncome')),
            CoapplicantIncome=float(request.form.get('CoapplicantIncome')),
            LoanAmount=float(request.form.get('LoanAmount')),
            Loan_Amount_Term=float(request.form.get('Loan_Amount_Term')),
            Credit_History=float(request.form.get('Credit_History')),
            Property_Area=request.form.get('Property_Area'),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction",pred_df.info())

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction",results)
        if results[0]==1.0:
            loan='Loan_Status:Yes'
        else:
            loan='Loan_Status:No'
        return render_template('home.html',results=loan)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        

