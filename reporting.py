import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])
output_path = os.path.join(config['output_model_path'])

# Function for reporting
def score_model():
    _, y = diagnostics.read_data(os.path.join(os.getcwd(), test_data_path), 'testdata.csv')
    pred = diagnostics.model_predictions(os.path.join(os.getcwd(), test_data_path), 'testdata.csv')
    disp = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(pred, y))
    disp.plot()
    fig = disp.figure_
    fig.savefig(os.path.join(os.getcwd(), output_path, 'confusionmatrix.png'))

if __name__ == '__main__':
    score_model()
