import requests
import json

# URL for the GET request
url = "https://datasets-server.huggingface.co/rows?dataset=Anthropic%2Fhh-rlhf&config=default&split=train&offset=0&length=100"

# Send the GET request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Parse the JSON response

    # Write data to a JSON file
    with open('./dataset_raw.json', 'w') as file:
        json.dump(data, file)
    print("Data has been written to dataset.json")
else:
    print("Failed to retrieve data. Status code:", response.status_code)

data_list = data['rows']

# Extract only the necessary information from each row
rows_list = [{"chosen": row['row']['chosen'], "rejected": row['row']['rejected']} for row in data_list]

# Write data to a JSON file
with open('./dataset_rows.json', 'w') as file:
    json.dump(rows_list, file)
print("Data has been written to rows.json")