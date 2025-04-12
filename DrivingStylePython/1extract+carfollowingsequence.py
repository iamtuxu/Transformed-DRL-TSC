'''
This code is used for extracting car-following sequences
'''


import pandas as pd


# Define the CSV file path
csv_file = 'data\original_data.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file, delimiter=',')


# Group by 'Vehicle_ID'
grouped_df = df.groupby('Vehicle_ID')

# List to hold data frames satisfying the conditions
satisfied_data_frames = []
#
# Apply the extract function to all DataFrames in grouped_df and save them in a list
for _, df_1 in grouped_df:
    # Find consecutive sequences that satisfy the conditions
    consecutive_sequences = []
    current_sequence = []
    sequence_index = 1  # Initialize sequence index

    for index, row in df_1.iterrows():
        if row['Preceding'] != 0 and 0 < row['Space_Headway'] < 250:
            preceding_v_length = 0
            preceding_row = df[(df['Vehicle_ID'] == row['Preceding']) & (df['Global_Time'] == row['Global_Time'])]
            if not preceding_row.empty:
                preceding_v_length = preceding_row.iloc[0]['v_length']
            row['preceding_v_length'] = preceding_v_length
            current_sequence.append(row)
        else:
            if len(current_sequence) >= 120:
                # consecutive_sequences.append(pd.DataFrame(current_sequence))
                current_sequence_df = pd.DataFrame(current_sequence)
                current_sequence_df['Sequence_Index'] = sequence_index
                consecutive_sequences.append(current_sequence_df)
                sequence_index += 1  # Increment sequence index
            current_sequence = []

    # If the last sequence also satisfies the conditions, add it to the list
    if len(current_sequence) >= 120:
        # consecutive_sequences.append(pd.DataFrame(current_sequence))
        current_sequence_df = pd.DataFrame(current_sequence)
        current_sequence_df['Sequence_Index'] = sequence_index
        consecutive_sequences.append(current_sequence_df)
    print(row['Vehicle_ID'])

    # Add consecutive sequences to the final list if they meet the length requirement
    satisfied_data_frames.extend(consecutive_sequences)


if satisfied_data_frames:
    combined_df = pd.concat(satisfied_data_frames)
    combined_df.to_csv('data/data.csv', index=False)


#
