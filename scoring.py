import pandas as pd
import pickle
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 


# Function for model scoring
def score_model():

    #  load trained model
    model_dir = os.path.join(os.getcwd(), model_path)
    model = pickle.load(open(os.path.join(os.getcwd(), model_dir, 'trainedmodel.pkl'), 'rb'))

    # load test data 
    testdata_dir = os.path.join(os.getcwd(), test_data_path)
    testdata = pd.read_csv(os.path.join(testdata_dir, 'testdata.csv'))

    X_test = testdata[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = testdata['exited']

    # predict & f1-score 
    pred = model.predict(X_test)
    f1score = metrics.f1_score(pred, y_test)

    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as txt_file:
        txt_file.write(str(f1score))

if __name__ == '__main__':
    score_model()
