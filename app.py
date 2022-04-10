from flask import Flask,request,render_template
import joblib
import numpy as np
app=Flask(__name__)
model=joblib.load("model")
label1 =joblib.load("mar_transform")
label2 =joblib.load("res_transform")
column= joblib.load("column")

app= Flask(__name__)

@app.route('/')
def predict():
    return render_template("index.html")

@app.route('/y_predict',methods=['POST'])
def y_predict():
    x_test=[[(x) for x in request.form.values()]]
    print('actual',x_test)
    x_test=np.array(x_test)
    x_test[:,4]=label1.transform(x_test[:,4])
    x_test[:,6]=label2.transform(x_test[:,6])
    x_test=column.transform(x_test)
    pred=model.predict(x_test)
    print(pred)
    if(pred[0]==0):
        result="no chances of stroke"
    else:result="chances of stroke"

    return render_template('index.html',\
                           prediction_text=('There are ',result))    
#@app.route('/login')
#def admin():
   # return "Hello admin"

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)