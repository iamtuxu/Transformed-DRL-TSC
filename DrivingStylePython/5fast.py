'''
This code is used for comparing fast driving style recognition methods with benchmark models
'''

import pandas as pd
import numpy as np
from IDM import idm
from scipy.optimize import minimize
import time
import os
import collections

def rmse_eval(df_1, v0, d_min, a, b, T):
    # Calculate 'Local_Z' and 'Leader_Z' columns
    df_eval = df_1.tail(50)
    mean_squared_error = 0
    for index, row in df_eval.iterrows():
        if index % 5 == 0:
            v = row['Ego_V']
            v_l = row['Leader_V']
            d = row['Gap']
            acc = idm(v, v_l, d, v0, d_min, a, b, T)
            mean_squared_error += (acc - row['Ego_A'])**2
    mean_squared_error
    return mean_squared_error

def rmse_cluster(df_1, v0, d_min, a, b, T, timewindow):
    df_est = (df_1.head(50)).tail(timewindow)
    mean_squared_error = 0
    for index, row in df_est.iterrows():
        v = row['Ego_V']
        v_l = row['Leader_V']
        d = row['Gap']
        acc = idm(v, v_l, d, v0, d_min, a, b, T)
        mean_squared_error += (acc - row['Ego_A'])**2
    mean_squared_error /= timewindow
    return mean_squared_error


# Define the CSV file path
csv_file = 'data/test.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file, delimiter=',')
grouped_df = df.groupby(['Vehicle_ID', 'Sequence_Index'])

'''
这里是方法1
'''
######## from literature
params = [33.3, 1.6, 0.73, 1.67, 2.0]
v0, d_min, a, b, T = params
total_error = sum(rmse_eval(group, v0, d_min, a, b, T) for _, group in grouped_df)
print(total_error)

'''
这里是方法2
'''
params = [30.7, 0.27, 0.73, 1.67, 1.11]
v0, d_min, a, b, T = params
total_error = sum(rmse_eval(group, v0, d_min, a, b, T) for _, group in grouped_df)
print(total_error)

'''
这里是方法3,方法3与轨迹长度有关，不妨先设置成5秒，但把5秒作为一个变量。但我发现0.1秒就够了 效果最好
'''

start_time = time.time()


params_prototype = [[34.1, 0.18, 0.73, 1.67, 0.97],
                    [40.0, 2.66, 0.73, 1.67, 0.70],
                    [40.0, 0.90, 0.73, 1.67, 1.45],
                    [10.6, 4.26, 0.73, 1.67, 0.30]]
timewindow = 1
total_error = 0
# List to store best indices for each group
best_indices = []
for _, group in grouped_df:
    ## 先估计 到底是哪一组
    # 初始化变量以找到最小的 RMSE 和对应的参数
    min_rmse = float('inf')
    best_params = None
    best_index = -1  # Initialize best_index to an invalid index

    # Iterate through params_prototype with enumeration to keep track of the index
    for index, params in enumerate(params_prototype):
        v0, d_min, a, b, T = params
        current_rmse = rmse_cluster(group, v0, d_min, a, b, T, timewindow)
        if current_rmse < min_rmse:
            min_rmse = current_rmse
            best_params = params
            best_index = index  # Update best_index when a new min_rmse is found
    # Store the index of the best parameters
    best_indices.append(best_index)
    v0, d_min, a, b, T = best_params
    total_error += rmse_eval(group, v0, d_min, a, b, T)
# print(total_error)
# index_count = collections.Counter(best_indices)
# print(best_indices)
# print(index_count)

    # # 输出最佳参数和最小的 RMSE
    # print("Best parameters for minimum RMSE:", best_params)
    # print("Minimum RMSE achieved:", min_rmse)
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
'''
这里是方法4,方法4也与轨迹长度有关，不妨先设置成5秒，但把5秒作为一个变量。
'''

start_time = time.time()
def objective(params):
    v0, d_min, T = params
    return rmse_cluster(group, v0, d_min, 0.73, 1.67, T, timewindow)


total_error = 0
for _, group in grouped_df:
    ## 先估计 到底是哪一组
    # 初始化变量以找到最小的 RMSE 和对应的参数
    # Initial guess for parameters
    initial_guess = [26.7, 0.3, 1.5]

    # Define lower and upper bounds for parameters
    lower_bounds = [10, -1, 0.3]  # Example lower bounds
    upper_bounds = [40, 10, 3]  # Example upper bounds

    result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=list(zip(lower_bounds, upper_bounds)),
                       options={'fatol': 0.05, 'xatol': 1})
    v0, d_min, T = result.x
    # print(result.x)
    total_error += rmse_eval(group, v0, d_min, 0.73, 1.67, T)
print(total_error)
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)