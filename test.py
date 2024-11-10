import pickle
import json
import pandas as pd
# with open("./project/output/posting_lists.pkl", "rb") as f:
#     posting_list = pickle.load(f)
#     data = json.loads(posting_list)  

# with open("./posting_lists.json", "w") as json_file:
#     json.dump(data, json_file, indent=4) 



data = pd.read_pickle("./project/output/gps_map_useful_method2.pkl")
print(data)