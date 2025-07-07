# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:44:42 2024

@author: sidra
"""

import os
import pandas as pd

# Path to the dataset
dataset_path = r'C:\Users\sidra\OneDrive\Desktop\d-final-face\dataSet'

# Initialize an empty list to store file paths and labels
data = []

# Traverse the directory structure
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):  # Add more extensions if needed
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            data.append([file_path, label])

# Create a DataFrame from the list
df = pd.DataFrame(data, columns=['file_path', 'label'])

# Save the DataFrame to a CSV file
output_csv = r'C:\Users\sidra\OneDrive\Desktop\d-final-face\dataset_labels.csv'
df.to_csv(output_csv, index=False)

print(f"CSV file has been created at: {output_csv}")
