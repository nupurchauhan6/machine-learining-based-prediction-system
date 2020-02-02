from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('data/mlrmodel.pkl','rb'))


@app.route('/')
def home():
   return render_template('form.html')

@app.route('/analysis',methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        AM_TH= request.form['AM_TH']
        
        AM_TT = request.form['AM_TT']
        AM_TW = request.form['AM_TW']
        AP_TH= request.form['AP_TH']
        AP_TT = request.form['AP_TT']
        AP_TW = request.form['AP_TW']
        AC_TH= request.form['AC_TH']
        AC_TT = request.form['AC_TT']
        AC_TW = request.form['AC_TW']
        M_TH= request.form['M_TH']
        M_TT = request.form['M_TT']
        M_TW = request.form['M_TW']
        M_V = request.form['M_V']
        BEE_TH= request.form['BEE_TH']
        
        BEE_TT = request.form['BEE_TT']
        BEE_TW = request.form['BEE_TW']
        BEE_V = request.form['BEE_V']
        EVS_TH= request.form['EVS_TH']
        EVS_TT = request.form['EVS_TT']
        W_TW = request.form['W_TW']
      
        
        
        sample_data = [AM_TH,AM_TT,AM_TW,AP_TH,AP_TT,AP_TW,AC_TH,AC_TT, AC_TW,M_TH,M_TT,M_TW, M_V,BEE_TH,BEE_TT, BEE_TW,BEE_V,EVS_TH,EVS_TT,W_TW]
        clean_data = [float(i) for i in sample_data]
        ex = np.array(clean_data).reshape(1,-1)
        result_prediction = model.predict(ex)

        return render_template('analysis.html',result_prediction=result_prediction,AM_TH=AM_TH,AM_TW=AM_TW,AM_TT=AM_TT,AC_TH=AC_TH,AC_TW=AC_TW,AC_TT=AP_TT,AP_TH=AP_TH,AP_TW=AP_TW,AP_TT=AP_TT,M_TH=M_TH,M_TW=M_TW,M_TT=M_TT,M_V=M_V,BEE_TH=BEE_TH,BEE_TW=BEE_TW,BEE_TT=BEE_TT,BEE_V=BEE_V,EVS_TT=EVS_TT,EVS_TH=EVS_TH,W_TW=W_TW)
        


if __name__ == '__main__':
    app.run(port=4000, debug=True)
