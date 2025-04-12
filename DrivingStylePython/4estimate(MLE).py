import pandas as pd
import numpy as np
from IDM import idm
from scipy.optimize import minimize
import time

def logL(df_1, v0, d_min, a, b, T, sigma):
    # Calculate 'Local_Z' and 'Leader_Z' columns
    log_likelihood = 0
    for index, row in df_1.iterrows():
        v = row['Ego_V']
        v_l = row['Leader_V']
        d = row['Distance']
        acc = idm(v, v_l, d, v0, d_min, a, b, T)
        log_likelihood += -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((acc - row['Ego_A']) / sigma)**2
    return log_likelihood

# Define a callback function to display intermediate results and check termination conditions
class CallbackState:
    def __init__(self):
        self.previous_LL = float('inf')
        self.consecutive_steps = 0

def callback(params):
    current_LL = objective(params)
    print("Parameters:", params)
    print("Total Likelihood:", current_LL)
    print()

    with open('optimize+result/optimization_results.txt', 'a') as file:
        file.write(f"Parameters: {params}\n")
        file.write(f"Total Log Likelihood: {current_LL}\n\n")

    # Check for termination conditions
    if abs(current_LL - callback_state.previous_LL) < 1e-6:
        callback_state.consecutive_steps += 1
    else:
        callback_state.consecutive_steps = 0

    if callback_state.consecutive_steps >= 5:
        print("Convergence achieved. Terminating optimization.")
        return True

    callback_state.previous_LL = current_LL
    return False

## 总体计算
# # Define the CSV file path
# csv_file = 'data/train.csv'
#
# # Read the CSV file into a pandas DataFrame
# df = pd.read_csv(csv_file, delimiter=',')
#
#
# # Group by 'Vehicle_ID' and 'Sequence Index'
# grouped_df = df.groupby(['Vehicle_ID', 'Sequence_Index'])
#
#
# # Define the objective function to minimize
# #### 这个是shared across the train dataset
# def objective(params):
#     v0, d_min, a, b, T = params
#     total_error = sum(rmse(group, v0, d_min, a, b, T) for _, group in grouped_df)
#     return total_error

# Define the CSV file path
csv_file = 'data/train_cluster.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file, delimiter=',')
df0 = df[df['cluster'] == 0]
df1 = df[df['cluster'] == 1]
df2 = df[df['cluster'] == 2]

# Group by 'Vehicle_ID' and 'Sequence Index'
grouped_df = df.groupby(['Vehicle_ID', 'Sequence_Index'])


# Define the objective function to minimize
#### 这个是shared across the train dataset
def objective(params):
    v0, d_min, a, b, T, sigma = params
    total_LL = sum(logL(group, v0, d_min, a, b, T, sigma) for _, group in grouped_df)
    return -total_LL


start_time = time.time()



# Initial guess for parameters
initial_guess = [40, 3, 1, 3, 2, 1]

# Define lower and upper bounds for parameters
lower_bounds = [15, 0, 0.5, 1, 0.5, 0.01]  # Example lower bounds
upper_bounds = [50, 15, 2, 4, 5, 2]       # Example upper bounds


# Perform optimization with bounds
callback_state = CallbackState()
result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=list(zip(lower_bounds, upper_bounds)), callback=callback)

# Extract the optimized parameters
v0_opt, d_min_opt, a_opt, b_opt, T_opt = result.x

print("Optimized parameters:")
print("v0:", v0_opt)
print("d_min:", d_min_opt)
print("a:", a_opt)
print("b:", b_opt)
print("T:", T_opt)

# Calculate the total mean squared error using the optimized parameters
total_error = objective(result.x)
print("Total mean squared error:", total_error)

end_time = time.time()
execution_time = end_time - start_time
print("结束！","运行时长：",execution_time)



