# Earthquake Physics & Machine Learning Project

**Authors:** Alessandro Cattaneo & Alessandra Melchionna

A Deep Learning project focused on seismic events using the INGV Instance Dataset. This repository contains the preprocessing logic, the U-Net training pipeline, and a live inference agent.

## 📁 Files
*   `code/EQ_Physics_Project.ipynb`: Main notebook for Training & Architecture.
*   `code/live_seismic_agent.py`: The AI Agent for real-time inference.
*   `code/preprocess_data.py`: Data cleaning and generation scripts.
*   `code/requirements.txt`: Python dependencies.
*   `CattaneoMelchionnaProjectProposal.docx` & `DEEP_MelchionnaCattaneo.pdf`: Project proposal.
*   `Report.pdf` & `Final_Presentation.pptx`: Final documentation.

## 🛠 Instructions

1.  **Download Data:** Get the [INGV Instance Dataset (Sample v3)](https://github.com/INGV/instance).
2.  **Preprocess:** Run `preprocess_data.py` to create `seismic_dataset_balanced.npz`.
3.  **Train:** Run the `EQ_Physics_Project.ipynb` notebook to generate `best_unet_model.pth`.
4.  **Run Agent:** Execute `seismic_agent.py` to get a real-time report of an area.

---
**Disclaimer:** This software is for research/educational purposes only. Do not use for safety-critical applications.
