# Experiment 1: Driving Style Analysis  

1. **Data Preprocessing: [`1extract+carfollowingsequence.py`]**  

   - Responsible for data denoising and extraction of car-following pairs from the Peachtree dataset.  

2. **Feature Extraction: [`2calculate+momentum.py`]**  

   - Extracts features from trajectory data.  

3. **Clustering Analysis: [`3PCA+kmeans.py`]**  

   - Performs clustering analysis using Principal Component Analysis (PCA) followed by K-means clustering.  

4. **Offline Parameter Estimation: [`4esimate(3 parameter).py`]**  

   - Involves estimating three parameters offline to form the foundational model for analysis.  

5. **Online Analysis: [`5fast.py`]**  

   - Conducts online analysis to enable real-time insights and data adjustments.  

6. **Data Visualization: [`6plotPaper.py`]**  

   - Visualizes results for reporting and paper publication.  

**Intelligent Driver Model Simulation: [`IDM.py`]**  

   - Simulates the Intelligent Driver Model.  

## Results:  
Please run the code following the order from 1 to 6.  

# Experiment 2: Traffic Signal Control  

We perform simulation experiments on various DRL-based traffic signal control systems to analyze the advantages of incorporating the proposed RL state parameter. This approach aims to enhance the effectiveness and efficiency of traffic signal controls through real-time adaptive strategies.  

## Run Code:  

Please install SUMO and Python first. Then install the required packages and run `main.py`.  


### Required Packages:  
- `absl-py==1.3.0`  
- `torch==2.0.1`  
- `numpy==1.24.1`  
- `gym==0.19.0`  
- `matplotlib==3.7.1`  

## Results:  
After running, results will be stored at avg-queue.txt for training cureve generation.
For evaluation files, avereage queue length will be printed after each simulation.

## Paper Link  
- *Real-Time Driving Style Integration in Deep Reinforcement Learning for Traffic Signal Control* (Under Review)  

## References  

- **[Traffic Light RL GitHub Repository](https://github.com/Desny/traffic_light_rl)**  
  - Y. Xu, Y. Wang, and C. Liu, "Training a reinforcement learning agent with autorl for traffic signal control," presented at the 2022 Euro-Asia Conference on Frontiers of Computer Science and Information Technology (FCSIT), 2022, pp. 51–55.  

- **Traffic Light Control Using Deep Policy-Gradient and Value-Function-Based Reinforcement Learning**  
  - S. S. Mousavi, M. Schukat, and E. Howley, "Traffic light control using deep policy-gradient and value-function-based reinforcement learning," IET Intelligent Transport Systems, vol. 11, no. 7, pp. 417–423, 2017.  

- **Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control**  
  - T. Chu, J. Wang, L. Codecà, and Z. Li, "Multi-agent deep reinforcement learning for large-scale traffic signal control," IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 3, pp. 1086–1095, 2020.
