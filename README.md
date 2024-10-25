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

# Experiment 2: Traffic Signal Control  

We perform simulation experiments on various DRL-based traffic signal control systems to analyze the advantages of incorporating the proposed RL state parameter. This approach aims to enhance the effectiveness and efficiency of traffic signal controls through real-time adaptive strategies.  

## Project Structure  

- **Agents:**  
  - The DRL agent responsible for learning optimal traffic signal strategies.  

- **Environment:**  
  - Includes the SUMO simulation and the traffic signal controller for realistic traffic scenario simulations.  

- **Nets:**  
  - Contains the traffic network and traffic flow data necessary for simulation.  

- **Record:**  
  - Stores the average queue length monitored during training along with generated plots for analysis.  

- **Weights:**  
  - Stores the weight matrix from training, used for evaluating and deploying trained models.  

## Run Code:  

Please install SUMO and Python first. Then install the required packages and run `main.py`.  

### Required Packages:  
- `absl-py==1.3.0`  
- `torch==2.0.1`  
- `numpy==1.24.1`  
- `gym==0.19.0`  
- `matplotlib==3.7.1`  

### `main.py` Configuration  

- **Simulation Time**  
  - `flags.DEFINE_float('simulation_time', 8001, 'time for simulation')`  
  - Sets the duration for the simulation.  

- **Traffic Flow Data**  
  - `flags.DEFINE_string('route_file', 'nets/2way-single-intersection/train.rou.xml', '')`  
  - Specifies the route file for traffic flow data.  

- **Use GUI**  
  - `flags.DEFINE_bool('use_gui', False, 'use sumo-gui instead of sumo')`  
  - Toggle to use `sumo-gui` for a graphical interface instead of the default `sumo`.  

- **Number of Episodes**  
  - `flags.DEFINE_integer('num_episodes', 201, '')`  
  - Defines the number of episodes for the simulation.  

- **Train or Evaluation Mode**  
  - `flags.DEFINE_string('mode', 'train', '')`  
  - Choose between 'train' or 'eval' modes.  

- **Weight Matrix for Evaluation**  
  - `flags.DEFINE_string('network_file', '', '')`  
  - Specify the weights file for policy evaluation (use `state2` for proposed and `state0` for conventional).  

### `traffic_signal.py` Configuration  

- **State Option**  
  - `self.state_option = 2` for the proposed method  
  - `self.state_option = 0` for the conventional method  

## Paper Link  
- *Real-Time Driving Style Integration in Deep Reinforcement Learning for Traffic Signal Control* (Under Review)  

## References  

- **[Traffic Light RL GitHub Repository](https://github.com/Desny/traffic_light_rl)**  
  - Y. Xu, Y. Wang, and C. Liu, "Training a reinforcement learning agent with autorl for traffic signal control," presented at the 2022 Euro-Asia Conference on Frontiers of Computer Science and Information Technology (FCSIT), 2022, pp. 51–55.  

- **Traffic Light Control Using Deep Policy-Gradient and Value-Function-Based Reinforcement Learning**  
  - S. S. Mousavi, M. Schukat, and E. Howley, "Traffic light control using deep policy-gradient and value-function-based reinforcement learning," IET Intelligent Transport Systems, vol. 11, no. 7, pp. 417–423, 2017.  

- **Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control**  
  - T. Chu, J. Wang, L. Codecà, and Z. Li, "Multi-agent deep reinforcement learning for large-scale traffic signal control," IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 3, pp. 1086–1095, 2020.
