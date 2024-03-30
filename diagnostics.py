
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
from importlib.metadata import version
from tabulate import tabulate

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

prod_path = os.path.join(config['prod_deployment_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_path = os.path.join(config['output_folder_path'])

# Function to read a csv file into a Pandas DataFrame
def read_data(path, file_name):
    testdata = pd.read_csv(os.path.join(path, file_name))
    X = testdata[['lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited']]
    y = testdata['exited']
    return X, y

# Function to get model predictions
def model_predictions(path, file_name):
    #read the deployed model
    model_dir = os.path.join(os.getcwd(), prod_path)
    model = pickle.load(open(os.path.join(os.getcwd(), model_dir, 'trainedmodel.pkl'), 'rb'))
    # and a test dataset
    X, _ = read_data(os.path.join(os.getcwd(), path), file_name)
    X.drop(['exited'], axis=1, inplace=True)
    # calculate predictions
    pred = list(model.predict(X))
    return pred

# Function to get summary statistics
def dataframe_summary(data_path, file_name):
    summary_stats = []
    X, _ = read_data(data_path, file_name)
    pred = model_predictions(data_path, file_name)
    X['prediction'] = pred
    for col in X.columns:
        summary_stats.append(X[col].mean())
        summary_stats.append(X[col].median())
        summary_stats.append(X[col].std())
    print(summary_stats)    
    return summary_stats

# Function to get N/A's
def dataframe_na(data_path, file_name):
    na_count = []
    X, _ = read_data(data_path, file_name)
    pred = model_predictions(data_path, file_name)
    X['prediction'] = pred
    for col in X.columns:
        na_count.append(X[col].isna().sum()/X.shape[0])
    print(na_count)   
    return na_count

##################Function to get timings
def execution_time():
    timing_list = []
    # timing ingestion script
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing_list.append(timeit.default_timer() - starttime)

    # timing training script
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing_list.append(timeit.default_timer() - starttime)
    print(timing_list)
    return timing_list

##################Function to check dependencies
def outdated_packages_list(requirements_file):
    # Read the requirements.txt file
    with open(requirements_file, 'r') as f:
        requirements = f.readlines()

    # Strip newline characters and split by '==' to get module names
    modules = [line.strip().split('==')[0] for line in requirements]
    versions = [line.strip().split('==')[1] for line in requirements]
    # Initialize a list to store module information
    module_info = []
    for module, vers in zip(modules, versions):
    # Append module information to the list
        module_info.append([module, vers, version(module)])

    # Output module information in a table format
    headers = ["Module Name", "Latest Version", "Installed Version"]
    table = tabulate(module_info, headers=headers, tablefmt="grid")
    print(table)
    return table


if __name__ == '__main__':
    model_predictions(os.path.join(os.getcwd(), test_data_path), 'testdata.csv')
    dataframe_summary(os.path.join(os.getcwd(), output_path), 'finaldata.csv')
    dataframe_na(os.path.join(os.getcwd(), output_path), 'finaldata.csv')
    execution_time()
    outdated_packages_list('requirements.txt')
