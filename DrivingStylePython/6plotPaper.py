import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
算法效果对比图
'''
# x = [0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5]
# y1 = [4883]*8
# y2 = [4167]*8
# y3 = [4401, 4450, 4651, 4672, 4552, 4479, 4548, 4328]
# y4 = [3838, 3858, 3883, 3871, 3858, 3900, 3860, 3915]
# plt.xlabel('trajectory duration (s)')
# plt.ylabel('SSE of the predicted and observed speeds')
# plt.plot(x, y1, label='benchmark method 1', color='#8D2F25')
# plt.plot(x, y2, label='benchmark method 2', color='#CB9475')
# plt.plot(x, y3, label='benchmark method 3',marker='+', color='#9467bd')
# plt.plot(x, y4, label='proposed method',marker='+',  color='#8CBF87')
# plt.legend(bbox_to_anchor=(0.55,0.95), loc='upper left')
# plt.show()

'''
算法时间对比图
'''
x = [0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5]
y3 = [1.0, 1.3, 1.8, 2.7, 3.7, 4.8, 7.5, 11.3]
y4 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.7]
plt.xlabel('trajectory duration (s)')
plt.ylabel('calculation time for driver parameter estimation (seconds)')
plt.plot(x, y3, label='benchmark method 3',marker='+', color='#9467bd')
plt.plot(x, y4, label='proposed method',marker='+',  color='#8CBF87')
plt.legend(bbox_to_anchor=(0.55,0.5), loc='upper left')
plt.show()

'''
Peachtree 轨迹
'''
# # Define the CSV file path
# csv_file = 'data/original_data.csv'
#
# # Read the CSV file into a pandas DataFrame
# df = pd.read_csv(csv_file, delimiter=',')
#
# # Filtering the DataFrame
# df = df[(df['Global_Time'] > 1163019100) & (df['Lane_ID'] == 0) & (df['Direction'] == 2) & (df['Movement'] == 1)]
#
# # Calculate the difference of Local_Y to find increasing segments
# df['Local_Y_Diff'] = df['Local_Y'].diff() * 3.048 * 3.6
# df['Local_Y'] = df['Local_Y'] * 0.3048
# df['Global_Time'] = (df['Global_Time'] - 1163019100) / 1000
# # Filter rows where Local_Y is increasing
# df = df[df['Local_Y_Diff'] > 0]
# df['Global_Time_Diff'] = df['Global_Time'].diff()
# df = df[df['Global_Time_Diff'] > 0]
# df = df[(df['Global_Time'] > 100) & (df['Global_Time'] < 700)]
#
# # Normalize the Local_Y_Diff for color mapping
# norm = plt.Normalize(0, 65)
# cmap = plt.get_cmap('RdYlGn')
#
# # Plotting
# fig, ax = plt.subplots(figsize=(15, 5))
# for vehicle_id, group in df.groupby('Vehicle_ID'):
#     # Create a color array for the line segments
#     colors = cmap(norm(group['Local_Y_Diff'].values))
#
#     # Plot each segment with a color corresponding to its speed change
#     for i in range(1, len(group)):
#         ax.plot(group['Global_Time'].iloc[i - 1:i + 1], group['Local_Y'].iloc[i - 1:i + 1], color=colors[i - 1])
#
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Local Y (m)')
# plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Speed (km/h)')
# plt.show()