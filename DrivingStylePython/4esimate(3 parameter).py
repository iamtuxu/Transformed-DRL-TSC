'''
This code is used for local parameter estimation from offline trajectories
'''

import pandas as pd
import numpy as np
from IDM import idm
from scipy.optimize import minimize
import time
import os

def rmse(df_1, v0, d_min, T):
    # Calculate 'Local_Z' and 'Leader_Z' columns
    mean_squared_error = 0
    for index, row in df_1.iterrows():
        if index % 2 == 0:
            v = row['Ego_V']
            v_l = row['Leader_V']
            d = row['Gap']
            acc = idm(v, v_l, d, v0, d_min, 0.73, 1.67, T)
            mean_squared_error += (acc - row['Ego_A'])**2
    mean_squared_error /= 50
    return mean_squared_error

# Define a callback function to display intermediate results and check termination conditions
class CallbackState:
    def __init__(self):
        self.previous_error = float('inf')
        self.consecutive_steps = 0

def callback(params):
    current_error = objective(params)
    print("Parameters:", params)
    print("Total mean squared error:", current_error)
    print()

    with open('optimize_result/optimization_results.txt', 'a') as file:
        file.write(f"Parameters: {params}\n")
        file.write(f"Total mean squared error: {current_error}\n\n")

    # Check for termination conditions
    if abs(current_error - callback_state.previous_error) < 1e-6:
        callback_state.consecutive_steps += 1
    else:
        callback_state.consecutive_steps = 0

    if callback_state.consecutive_steps >= 5:
        print("Convergence achieved. Terminating optimization.")
        return True

    callback_state.previous_error = current_error
    return False

# Define the CSV file path
csv_file = 'data/train_cluster.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file, delimiter=',')
clusters = df['cluster'].unique()

# 创建存储优化结果的文件夹
os.makedirs('optimize_result', exist_ok=True)


def objective(params):
    v0, d_min, T = params
    total_error = sum(rmse(group, v0, d_min, T) for _, group in grouped_df)
    return total_error


# 针对每个集群分别进行优化
for cluster in clusters:
    print(clusters)
    # 筛选出当前集群的数据
    cluster_df = df[df['cluster'] == cluster]
    print("cluster ID", cluster)
    # Group by 'Vehicle_ID' and 'Sequence Index'
    grouped_df = cluster_df.groupby(['Vehicle_ID', 'Sequence_Index'])

    # Define the objective function to minimize


    start_time = time.time()

    # Initial guess for parameters
    initial_guess = [26.7, 0.3, 1.5]

    # Define lower and upper bounds for parameters
    lower_bounds = [10, -1, 0.3]  # Example lower bounds
    upper_bounds = [40, 10, 3]  # Example upper bounds

    # Perform optimization with bounds
    callback_state = CallbackState()
    result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=list(zip(lower_bounds, upper_bounds)),
                      callback=callback, options={'fatol': 0.05, 'xatol': 1})

    # Extract the optimized parameters
    v0_opt, d_min_opt, T_opt = result.x

    print(f"Optimized parameters for cluster {cluster}:")
    print("v0:", v0_opt)
    print("d_min:", d_min_opt)
    print("T:", T_opt)

    # Calculate the total mean squared error using the optimized parameters
    total_error = objective(result.x)
    print(f"Total mean squared error for cluster {cluster}: {total_error}")

    # 保存结果到对应文件
    with open(f'optimize_result/optimization_results_cluster_{cluster}.txt', 'w') as file:
        file.write(f"Optimized parameters for cluster {cluster}:\n")
        file.write("v0: " + str(v0_opt) + "\n")
        file.write("d_min: " + str(d_min_opt) + "\n")
        file.write("T: " + str(T_opt) + "\n")
        file.write(f"Total mean squared error for cluster {cluster}: {total_error}\n")

    # 计算执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print("结束！","运行时长：",execution_time)


print("overall")
# Group by 'Vehicle_ID' and 'Sequence Index'
grouped_df = df.groupby(['Vehicle_ID', 'Sequence_Index'])
start_time = time.time()

# Initial guess for parameters
initial_guess = [26.7, 0.3, 1.5]

# Define lower and upper bounds for parameters
lower_bounds = [10, -1, 0.3]  # Example lower bounds
upper_bounds = [40, 10, 3]  # Example upper bounds

# Perform optimization with bounds
callback_state = CallbackState()
result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=list(zip(lower_bounds, upper_bounds)),
                      callback=callback, options={'fatol': 0.05, 'xatol': 1})

# Extract the optimized parameters
v0_opt, d_min_opt, T_opt = result.x

print(f"Optimized parameters for overall:")
print("v0:", v0_opt)
print("d_min:", d_min_opt)
print("T:", T_opt)

# Calculate the total mean squared error using the optimized parameters
total_error = objective(result.x)
print(f"Total mean squared error for overall: {total_error}")

# 保存结果到对应文件
with open(f'optimize_result/optimization_results_overall.txt', 'w') as file:
    file.write(f"Optimized parameters for overall:\n")
    file.write("v0: " + str(v0_opt) + "\n")
    file.write("d_min: " + str(d_min_opt) + "\n")
    file.write("T: " + str(T_opt) + "\n")
    file.write(f"Total mean squared error for overall: {total_error}\n")

# 计算执行时间
end_time = time.time()
execution_time = end_time - start_time
print("结束！","运行时长：",execution_time)