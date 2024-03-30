import json
import os
import subprocess
import training
import scoring
import deployment
import diagnostics
import reporting

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_path = os.path.join(config['prod_deployment_path'])
input_data_path = os.path.join(config['input_folder_path'])
output_data_path = os.path.join(config['output_folder_path'])

# Check and read new data
# first, read ingestedfiles.txt
ingestedfiles = open(os.path.join(os.getcwd(), prod_path, 'ingestedfiles.txt'), 'r').read().splitlines()

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
newfiles = os.listdir(os.path.join(os.getcwd(), input_data_path))

##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
proceed = set(ingestedfiles) - set(newfiles)
if proceed==False:
    print('No new data. Ending process.\n')
else:
    print('New data found!\n')
    # Run the ingestion process
    subprocess.run(["python", "ingestion.py"])
    # Score the new data
    newscore = scoring.score_model(data_path=output_data_path, data_file='finaldata.csv')
    # read the lastes score
    latestscore = float(open(os.path.join(os.getcwd(), prod_path, 'latestscore.txt'), 'r').read())
    # Checking for model drift
    if newscore > latestscore:
        print('The new score is greather than the latest. Therefore, there is no data drift.\n')
    else:
        print('Retraing...\n')
        # Re-deployment
        subprocess.run(["python", "training.py"])
        subprocess.run(["python", "deployment.py"])

# Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.run(["python", "diagnostics.py"])
subprocess.run(["python", "reporting.py"])
#subprocess.run(["python", "app.py", "run"])
os.system("python app.py &")
subprocess.run(["python", "apicalls.py"])
