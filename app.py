from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
import diagnostics
import scoring
#import predict_exited_from_saved_model
import json
import os

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

prediction_model = None

# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    pred = diagnostics.model_predictions(os.path.join(os.getcwd(), test_data_path), 'testdata.csv')
    return str(pred)

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    f1score = scoring.score_model(data_path=dataset_csv_path, data_file='finaldata.csv')
    return str(f1score)

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    summary_stats = diagnostics.dataframe_summary(os.path.join(os.getcwd(), dataset_csv_path), 'finaldata.csv')
    return str(summary_stats)

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics_info():        
    na_count = diagnostics.dataframe_na(os.path.join(os.getcwd(), dataset_csv_path), 'finaldata.csv')
    timing_list = diagnostics.execution_time()
    table = diagnostics.outdated_packages_list('requirements.txt')
    return jsonify({"na_count": na_count, "timing_list": timing_list, "table": table})

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)