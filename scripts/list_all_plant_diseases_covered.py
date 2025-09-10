import json
from tabulate import tabulate
import math
import csv

with open("data/plantvillage_kb.json", "r", encoding="utf-8") as f:
    kb = json.load(f)

# Set table width to half the max number of diseases (rounded up)
max_diseases = max(len(diseases) for diseases in kb.values())
table_width = math.ceil(max_diseases / 2)
headers = ["Plant"] + [f"Disease {i+1}" for i in range(table_width)]
rows = []

for plant, diseases in kb.items():
    disease_list = list(diseases.keys())
    # If 'healthy' exists, move it to the front
    if "healthy" in disease_list:
        disease_list.remove("healthy")
        disease_list = ["healthy"] + disease_list
    # Split diseases into chunks of table_width
    for i in range(0, len(disease_list), table_width):
        chunk = disease_list[i:i+table_width]
        row = [plant if i == 0 else ""] + chunk
        # Pad with empty strings if fewer diseases than table_width
        row += [""] * (table_width - len(chunk))
        rows.append(row)

print(tabulate(rows, headers=headers, tablefmt="grid"))
with open("plant_diseases_table.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    writer.writerows(rows)
