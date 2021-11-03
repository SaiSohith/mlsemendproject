from flask import Flask,render_template,request
import life_expectency_prediction_using_linear_regression as lxp
import pandas as pd
import numpy as np
import joblib
app = Flask(__name__)
# model = joblib.load('lifeexptencymodel.pkl')
@app.route('/',methods = ['GET','POST'])
def index():
    if request.method == 'GET': return render_template('abc.html')
    if request.method == 'POST':
        print("in post")
        a,b,c,d,e= request.form['a'],request.form['b'],request.form['c'],request.form['d'],request.form['e']
        print(a,b,c,d,e)
        data = {
            'Adult Mortality':[a] ,'infant deaths':[b] ,'Alcohol':[c] ,'Population':[d] ,'Education':[e]
        }
        X = pd.DataFrame(data)
        X = np.array(X).astype('float32')
        return render_template('abc.html',result=lxp.mod(X))
        

if __name__ == '__main__':
    app.run()