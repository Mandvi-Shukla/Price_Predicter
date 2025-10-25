import json
import joblib

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## Load the model
regmodel=joblib.load(open('trained_model.pkl','rb'))
ppPipeline=joblib.load(open('preprocessing_pipeline.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    # print("Incoming features:", data.keys())
    # print("Expected features:", ppPipeline.feature_names_in_)
    print(data)
    # print(np.array(list(data.values())).reshape(1,-1))
    # new_data=ppPipeline.transform(np.array(list(data.values())).reshape(1,-1))
    # output=regmodel.predict(new_data)
    input_df = pd.DataFrame([data])  # <- wraps your JSON data in a DataFrame
    new_data = ppPipeline.transform(input_df)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

#@app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     final_input=ppPipeline.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=regmodel.predict(final_input)[0]
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form values to floats
        data_list = [float(x) for x in request.form.values()]
        
        # Check if we have 13 features
        if len(data_list) != 13:
            return "Error: Expected 13 features, got {}".format(len(data_list))
        
         # Validate CHAS (categorical) - index 3 in Boston Housing
        chas_value = data_list[3]
        if chas_value not in [0, 1]:
            return "Error: CHAS must be 0 or 1"
        
        # Feature names for Boston Housing
        feature_names = [
            'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
            'DIS','RAD','TAX','PTRATIO','B','LSTAT'
        ]
        
        # Convert to DataFrame
        final_input = pd.DataFrame([data_list], columns=feature_names)
        
        # Apply preprocessing pipeline
        final_input_transformed = ppPipeline.transform(final_input)
        print(final_input_transformed)
        
        # Make prediction
        output = regmodel.predict(final_input_transformed)[0]
        
        # Return result to HTML
        return render_template("home.html", prediction_text=f"The House price prediction is {output}")
    
    except ValueError as e:
        return f"Invalid input: {e}"
if __name__=="__main__":
    app.run(debug=True)
   