import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import joblib
from collections.abc import Mapping

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))
columns = joblib.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    d = request.form.to_dict()
    d["Balance"]=int(d["Balance"])
    d["NoA"]=int(d["NoA"])
    d["Usia"]=int(d["Usia"])
    y=[]
    cat=["Gender", "Uker", 'income', 'pendidikan', 'sumber penghasilan']
    for _ in cat:
        y.append(d[_])
    
    df=pd.DataFrame(columns=columns)
    df.loc[df.shape[0]] = [0]*60
    df.Balance[0]=d["Balance"]
    df.NoA[0]=d["NoA"]
    df.Usia[0]=d["Usia"]
    for _ in df.columns:
        if _ in y:
            df[_][0]=1
    pred=model.predict_proba(df)
    pred=pred[0][1]
    kelas=""
    if (pred > 0 and pred < 0.4):
        kelas="Low Risk"
    elif (pred > 0.4 and pred < 0.7):
        kelas="Moderate Risk"        
    elif (pred > 0.7 and pred < 0.85):
        kelas="High Risk"
    else : kelas = "Very High Risk"

    return render_template('home.html', prediction_text='Kemungkinan user untuk churn adalah {}% sehingga masuk dalam kategori {} '.format(round(pred*100), kelas))


if __name__ == "__main__":
    app.run(debug=True)
