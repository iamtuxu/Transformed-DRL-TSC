# Transformed-DRL-TSC  

## Experiment 1: Driving Style Analysis  


1. **Data Preprocessing: [`1extract+carfollowingsequence.py`]**  

   - This script is responsible for data denoising and extraction of car-following pairs from the Peachtree dataset.  

2. **Feature Extraction: [`2calculate+momentum.py`]**  

   - Utilizing this script, features are extracted from the trajectory data.  

3. **Clustering Analysis: [`3PCA+kmeans.py`]**  

   - This step performs clustering analysis using Principal Component Analysis (PCA) followed by K-means clustering.  

4. **Offline Parameter Estimation: [`4esimate(3 parameter).py`]**  

   - This phase involves estimating three parameters offline, forming the foundational model for further analysis.  

5. **Online Analysis: [`5fast.py`]**  

   - Conducts the online part of the analysis, enabling real-time insights and data adjustments.  

6. **Data Visualization: [`6plotPaper.py`]**  

   - Visualizes the results for reporting and paper publication purposes.  

**Intelligent Driver Model Simulation: [`IDM.py`]**  

   - Simulates the Intelligent Driver Model.

## Experiment 2: Traffic Signal Control

We perform simulation experiments on various DRL-based traffic signal control systems to analyze the advantages of incorporating the proposed RL state parameter. This approach aims to enhance the effectiveness and efficiency of traffic signal controls through real-time adaptive strategies.  

## Project Structure  

For each project, the following components are included:  

- **Agents:**  
  - The DRL agent responsible for learning optimal traffic signal strategies.  

- **Environment:**  
  - Includes the SUMO simulation and the traffic signal controller that facilitate realistic traffic scenario simulations.  

- **Nets:**  
  - Contains the traffic network and traffic flow data necessary for simulation.  

- **Record:**  
  - Stores the average queue length monitored during the training process along with the generated plots for analysis.  

- **Weights:**  
  - Stores the weight matrix from the training process, used for evaluating and deploying trained models.  

## Paper Link  
- *Real-Time Driving Style Integration in Deep Reinforcement Learning for Traffic Signal Control* (Under Review)  

## References  
- **[Traffic Light RL GitHub Repository](https://github.com/Desny/traffic_light_rl)**  
  - Y. Xu, Y. Wang, and C. Liu, "Training a reinforcement learning agent with autorl for traffic signal control," presented at the 2022 Euro-Asia Conference on Frontiers of Computer Science and Information Technology (FCSIT), 2022, pp. 51–55.  

- **Traffic Light Control Using Deep Policy-Gradient and Value-Function-Based Reinforcement Learning**  
  - S. S. Mousavi, M. Schukat, and E. Howley, "Traffic light control using deep policy-gradient and value-function-based reinforcement learning," IET Intelligent Transport Systems, vol. 11, no. 7, pp. 417–423, 2017.  

- **Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control**  
  - T. Chu, J. Wang, L. Codecà, and Z. Li, "Multi-agent deep reinforcement learning for large-scale traffic signal control," IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 3, pp. 1086–1095, 2020. 

