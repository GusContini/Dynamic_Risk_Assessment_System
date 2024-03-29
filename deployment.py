import pickle
import os
import json
import shutil

# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])
ingested_data_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

# function for deployment
def store_model_into_pickle():
        
    # output directory
    output_dir = latestscore_dir = os.path.join(os.getcwd(), prod_deployment_path)

    # read latestscore.txt
    latestscore_dir = os.path.join(os.getcwd(), model_path)
    latestscore = open(os.path.join(os.getcwd(), latestscore_dir, 'latestscore.txt'), 'r').read()
    # write it to the output dir
    with open(os.path.join(os.getcwd(), output_dir, 'latestscore.txt'), 'w') as txt_file:
        txt_file.write(str(latestscore))

    # read ingestedfiles.txt
    ingested_dir = os.path.join(os.getcwd(), ingested_data_path)
    ingested_data = open(os.path.join(os.getcwd(), ingested_dir, 'ingestedfiles.txt'), 'r').read()
    # write it to the output dir
    with open(os.path.join(os.getcwd(), output_dir, 'ingestedfiles.txt'), 'w') as txt_file:
        txt_file.write(str(ingested_data))

    # copy trained model and save it to the output dir
    model_dir = os.path.join(os.getcwd(), model_path)
    shutil.copyfile(
        os.path.join(os.getcwd(), model_dir, 'trainedmodel.pkl'),
        os.path.join(os.getcwd(), output_dir, 'trainedmodel.pkl')
    )

if __name__ == '__main__':
    store_model_into_pickle()