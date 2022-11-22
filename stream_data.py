import requests
import json
import pandas as pd


instances = []
df = pd.read_csv("https://www.openml.org/data/get_csv/21792853/dataset")
expensive_df = df[df["price"] > 10000].sort_values("price", ascending=False)
df = df[df["price"] < 10000]

LOCAL_IP = "0.0.0.0:5050"
url = f"http://{LOCAL_IP}/diamonds/v1/predict"

count = 30
chunk_size = 500
reset_index = True
min_chunk, max_chunk = 0, chunk_size
while count:
    print(count)
    print(f"Uploading data from: {str(pd.Timestamp.now() - pd.Timedelta(count, 'd'))}")
    if count < 10:
        print(expensive_df.iloc[min_chunk:max_chunk]["price"].mean())
        if reset_index:
            min_chunk, max_chunk = 0, 500
            reset_index = False
        for row_tuple in expensive_df.iloc[min_chunk:max_chunk].iterrows():
            row_dict = row_tuple[1].drop("price").to_dict()
            row_dict["record_id"] = row_tuple[1].name
            row_dict["ts"] = str(pd.Timestamp.now() - pd.Timedelta(count, "d"))
            instances.append(row_dict)
    else:
        print(df.iloc[min_chunk:max_chunk]["price"].mean())
        for row_tuple in df.iloc[min_chunk:max_chunk].iterrows():
            row_dict = row_tuple[1].drop("price").to_dict()
            row_dict["record_id"] = row_tuple[1].name
            row_dict["ts"] = str(pd.Timestamp.now() - pd.Timedelta(count, "d"))
            instances.append(row_dict)

    response = requests.post(url, json=instances)
    print("---" * 15)
    instances = []
    count -= 1
    min_chunk += chunk_size
    max_chunk += chunk_size
