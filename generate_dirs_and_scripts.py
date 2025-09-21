#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a series of directories with simulation.py tailored for a given
parametrisation.

The file `simulation.py` HAS TO contain the line:

    pp = generate_pSI_selected_eff([eta], [delta], p_base)

otherwise, this script won't work as intended

@author: ChatGPT-4o and Piotr Bentkowski
"""
import os
import csv

# File paths
dir_file = "SBM_param_dirr_list.txt"         # params and directory names
simulation_template_file = "simulation.py"  # code of the simulation

# Read the CSV file to get eta, delta, and directory names
with open(dir_file, 'r') as f:
    reader = csv.reader(f)
    dir_entries = [{"eta": float(row[0]), "delta": float(row[1]),
                    "dir_name": row[2]} for row in reader]

# Read the template simulation.py file
with open(simulation_template_file, 'r') as f:
    simulation_template_lines = f.readlines()

# Process each directory and create a customized simulation.py
for entry in dir_entries:
    eta = entry["eta"]
    delta = entry["delta"]
    dir_name = entry["dir_name"]

    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Prepare the customized content
    customized_lines = []
    for line in simulation_template_lines:
        # Replace the placeholders in the target line
        if "pp = generate_pSI_selected_eff" in line:
            line = line.replace("eta", str(eta)).replace("delta", str(delta))
        customized_lines.append(line)

    # Write the customized simulation.py to the directory
    customized_file_path = os.path.join(dir_name, simulation_template_file)
    with open(customized_file_path, 'w') as sim_file:
        sim_file.writelines(customized_lines)

print("Directories and customized simulation.py files have been created!")
