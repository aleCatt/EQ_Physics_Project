# EQ_Physics_Project
Final project for the "Earthquake Physics and Machine Learning" course.
Made by Alessandro Cattaneo & Alessandra Melchionna.

Contents & usage:
- Download the data from the INGV Instance Dataset (Sample dataset version 3). https://github.com/INGV/instance
- Run the "preprocess_data.py" script obtaining "seismic_dataset_balanced.npz".
- Run the training pipeline present in "EQ_Physics_Project.ipynb" and save "best_unet_model.pth".
- Run the "seismic_agent.py" script in order to see a real time report of a desired area.

Disclaimer
This software is for research and educational purposes only. It should not be used as a primary warning system for safety-critical seismic events. Always refer to official sources (INGV, USGS) for earthquake information.
