import csv
import re
import requests
from collections import defaultdict


def get_html(source):
    # Detect if source is a URL or a local file path
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source)
        response.raise_for_status()
        return response.text
    else:
        with open(source, encoding='utf-8') as f:
            return f.read()


# Provide either a URL or a local file path here
# dataset_source = r'https://www.kaggle.com/datasets/alinedobrovsky/plant-disease-classification-merged-dataset'
dataset_source = r'c:\Users\msi\Downloads\Plant Disease Classification Merged Dataset.html'


html = get_html(dataset_source)

# Regex to match plant__disease or plant___disease (handles single/double underscores)
matches = re.findall(r'>([A-Za-z0-9_()\'\-]+__[_A-Za-z0-9()\'\-]+)<', html)

# Remove duplicates and sort
classes = sorted(set(matches))

# Group by plant and summarize
plant_diseases = defaultdict(list)

for cls in classes:
    # Split by double underscore to separate plant and disease
    plant, disease = cls.split('__', 1)
    plant_diseases[plant].append(disease)

print("\nSummary by plant:")
for plant, diseases in plant_diseases.items():
    healthy = [d for d in diseases if 'healthy' in d.lower()]
    disease_list = [d for d in diseases if 'healthy' not in d.lower()]
    disease_str = ', '.join(disease_list)
    healthy_str = f" + {len(healthy)} healthy" if healthy else ""
    print(f"{plant}: {disease_str}{healthy_str}")

# Optionally, write to a CSV
# with open('kaggle_classes.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Plant__Disease'])
#     for cls in classes:
#         writer.writerow([cls])

# Print or save to file
print(f"Total classes found: {len(classes)}")  # <-- Added count print
# for cls in classes:
#     print(cls)
