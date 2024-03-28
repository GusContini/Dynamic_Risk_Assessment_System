import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# Function for data ingestion
def merge_multiple_dataframe():
    
    # Create a Pandas DataFrame object
    main_df = pd.DataFrame(columns=[
        'corporation', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited'
        ])
    # Create a list to register ingested csv files
    ingested_files = []

    # check for datasets, compile them together, and write to an output file
    input_dir = os.path.join(os.getcwd(), input_folder_path)
    input_files = os.listdir(input_dir)
    for file_name in input_files:
        if '.csv' in file_name:
            df = pd.read_csv(os.path.join(input_dir,file_name))
            main_df = pd.concat([main_df, df], ignore_index=True)
            ingested_files.append(file_name)
    
    # drop dups
    main_df = main_df.drop_duplicates()

    # save DataFrame as a csv file to outout folder
    output_dir = os.path.join(os.getcwd(), output_folder_path)
    main_df.to_csv(os.path.join(output_dir, 'finaldata.csv'))
    # and now the ingested list as a txt file
    with open(os.path.join(output_dir, 'ingestedfiles.txt'), 'w') as tfile:
        tfile.write('\n'.join(ingested_files))

if __name__ == '__main__':
    merge_multiple_dataframe()
