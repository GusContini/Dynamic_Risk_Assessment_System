import requests
import json
import os

with open('config.json','r') as f:
    config = json.load(f) 

output_path = os.path.join(config['output_model_path'])

results = {}

# Call each API endpoint and store the responses
results['prediction'] = requests.post('http://127.0.0.1:8000/prediction?').content
results['scoring'] = requests.get('http://127.0.0.1:8000/scoring?').content
results['summarystats'] = requests.get('http://127.0.0.1:8000/summarystats?').content
diagnostics = requests.get('http://127.0.0.1:8000/diagnostics?')
data = diagnostics.json()
results['diagnostics - na_count'] = data["na_count"]
results['diagnostics - timing_list'] = data["timing_list"]
results['diagnostics - table'] = data["table"]

# Combine the results into a single string
combined_results = "\n\n".join([f"--- {endpoint.upper()} ---\n{result}" for endpoint, result in results.items()])

# Write the responses to your workspace
with open(os.path.join(os.getcwd(), output_path, 'apireturns.txt'), "w") as file:
    file.write(combined_results)
