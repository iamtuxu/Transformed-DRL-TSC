# Transformed-DRL-TSC  

## Experiment 1: Driving Style Analysis  

This experiment showcases the methodology for analyzing driving styles using DRL-TSC. Below are the detailed steps:  

1. **Data Preprocessing: [`extract+carfollowingsequence.py`]**  

   - This script is responsible for data denoising and extraction of car-following pairs from the Peachtree dataset.  

2. **Feature Extraction: [`calculate+momentum.py`]**  

   - Utilizing this script, features are extracted from the trajectory data.  

3. **Clustering Analysis: [`PCA+kmeans.py`]**  

   - This step performs clustering analysis using Principal Component Analysis (PCA) followed by K-means clustering.  

4. **Offline Parameter Estimation: [`esimate(3 parameter).py`]**  

   - This phase involves estimating three parameters offline, forming the foundational model for further analysis.  

5. **Online Analysis: [`fast.py`]**  

   - Conducts the online part of the analysis, enabling real-time insights and data adjustments.  

6. **Data Visualization: [`plotPaper.py`]**  

   - Visualizes the results for reporting and paper publication purposes.  

7. **Reinforcement Learning Plotting: [`RLplot.py`]**  

   - This script is aimed at plotting results specific to the reinforcement learning aspect of the study.  

8. **Intelligent Driver Model Simulation: [`IDM.py`]**  

   - Simulates the Intelligent Driver Model, providing insights into driver behavior under various conditions.
