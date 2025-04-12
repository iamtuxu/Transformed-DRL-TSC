'''
This code is used for creating training and testing data
'''

import pandas as pd
import numpy as np
import random

# Fix the random seed for reproducibility
random.seed(42)

# Define the CSV file path
csv_file = 'data\data.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file, delimiter=',')


# Group by 'Vehicle_ID'
grouped_df = df.groupby(['Vehicle_ID', 'Sequence_Index'])

def transform(df_1):
    # Calculate 'Local_Z' and 'Leader_Z' columns
    x_first_row = df_1.iloc[0]['Global_X']
    y_first_row = df_1.iloc[0]['Global_Y']
    df_1['Local_Z'] = np.sqrt((df_1['Global_X'] - x_first_row) ** 2 + (df_1['Global_Y'] - y_first_row) ** 2)
    df_1['Leader_Z'] = df_1['Local_Z'] + df_1['Space_Headway']

    # Convert 'Local_Z' and 'Leader_Z' values from feet to meters
    df_1[['Local_Z', 'Leader_Z', 'Space_Headway', 'preceding_v_length']] *= 0.3048
    df_1['Gap'] = df_1['Space_Headway'] - df_1['preceding_v_length']

    # Select relevant columns
    df_selected = df_1[['Vehicle_ID', 'Frame_ID', 'Space_Headway', 'Gap', 'Local_Z', 'Leader_Z', 'Sequence_Index']]


    # # Calculate 'v' and 'Leader_v' columns
    df_selected['Ego_V'] = (df_selected['Local_Z'].shift(-5) - df_selected['Local_Z'].shift(+5)).copy()
    df_selected['Leader_V'] = (df_selected['Leader_Z'].shift(-5) - df_selected['Leader_Z'].shift(+5)).copy()
    # 计算前十行的平均速度
    df_selected['Ego_V_prev_avg'] = df_selected['Ego_V'].rolling(window=10, min_periods=1).mean().shift(10)

    # 计算后十行的平均速度
    df_selected['Ego_V_next_avg'] = df_selected['Ego_V'].rolling(window=10, min_periods=1).mean().shift(-10)

    # 计算加速度：后十行平均速度减去前十行平均速度
    df_selected['Ego_A'] = df_selected['Ego_V_next_avg'] - df_selected['Ego_V_prev_avg']

    # 清理不再需要的列
    df_selected.drop(['Ego_V_prev_avg', 'Ego_V_next_avg'], axis=1, inplace=True)
    df_selected['Ego_A'] = (df_selected['Ego_V'].shift(-5) - df_selected['Ego_V'].shift(+5)).copy()
    # Fill empty values in 'Ego_V' column with the value from the last row
    df_selected['Ego_A'] = df_selected['Ego_A'].fillna(method='ffill')

    return df_selected.iloc[10:110]

#
# def transform(df_1):
#     # Calculate 'Local_Z' and 'Leader_Z' columns
#     x_first_row = df_1.iloc[0]['Global_X']
#     y_first_row = df_1.iloc[0]['Global_Y']
#     veh_length = df_1.iloc[0]['v_length']
#     print(veh_length)
#     df_1['Local_Z'] = np.sqrt((df_1['Global_X'] - x_first_row) ** 2 + (df_1['Global_Y'] - y_first_row) ** 2)
#     df_1['Leader_Z'] = df_1['Local_Z'] + df_1['Space_Headway']
#
#     # Convert 'Local_Z' and 'Leader_Z' values from feet to meters
#     df_1[['Local_Z', 'Leader_Z', 'Space_Headway']] *= 0.3048
#     df_1['distance'] = df_1['Space_Headway'] - veh_length * 0.3048
#
#     # Select relevant columns
#     df_selected = df_1[['Vehicle_ID', 'Frame_ID', 'Space_Headway', 'distance','Local_Z', 'Leader_Z', 'Sequence_Index']]
#
#
#     # # Calculate 'v' and 'Leader_v' columns
#     df_selected['Ego_V'] = 2 * (df_selected['Local_Z'].shift(-5) - df_selected['Local_Z'])
#     df_selected['Leader_V'] = 2 * (df_selected['Leader_Z'].shift(-5) - df_selected['Leader_Z'])
#     df_selected['Ego_A'] = 2 * (df_selected['Ego_V'].shift(-5) - df_selected['Ego_V'])
#     # Fill empty values in 'Ego_V' column with the value from the last row
#     df_selected['Ego_A'] = df_selected['Ego_A'].fillna(method='ffill')
#     return df_selected.iloc[5:105]

modified_sequences = [transform(group) for _, group in grouped_df]

# Randomly select 80% of the elements from modified_sequences
num_sequences_to_select = int(len(modified_sequences) * 0.8)
selected_indices = random.sample(range(len(modified_sequences)), num_sequences_to_select)

# Extract selected sequences and remaining sequences
selected_sequences = [modified_sequences[i] for i in selected_indices]
remaining_sequences = [seq for i, seq in enumerate(modified_sequences) if i not in selected_indices]

# Concatenate the selected DataFrames into a single DataFrame
concatenated_df = pd.concat(selected_sequences)

# Save the concatenated DataFrame as a CSV file
concatenated_df.to_csv('data/train.csv', index=False)

# Concatenate the remaining DataFrames into a single DataFrame
remaining_concatenated_df = pd.concat(remaining_sequences)

# Save the remaining concatenated DataFrame as a separate CSV file
remaining_concatenated_df.to_csv('data/test.csv', index=False)


# grouped_df = pd.concat(modified_sequences)

# # Save the combined DataFrame to a single CSV file
# combined_df.to_csv('data/sequences.csv', index=False)

# def rmse(df_1, v0, d_min, a, b, T):
#     ## select 5second of trajectory
#     selected_rows = df_1.iloc[10:510]
#     for index, row in selected_rows.iterrows():
#         v = row['Ego_v']
#         a = row['Ego_a']
#         print(v, a)


# rmse(extracted_dfs[0], 20, 3, 1, 1, 1)
# v = 20  # Current speed (m/s)
# v0 = 30  # Desired speed (m/s)
# v_l = 30  # leader_speed
# d = 10  # Gap to the vehicle in front (m)
# d_min = 7
# a = 1  # Maximum acceleration (m/s^2)
# b = 4  # Comfortable deceleration (m/s^2)
# T = 1  # Desired time headway (s)
#
# # Calculate acceleration using IDM
# acc = idm(v, v0, v_l, d, d_min, a, b, T)
# # print("Acceleration:", acc)
