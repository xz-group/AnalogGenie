import numpy as np
import os

# Directories to process
base_dirs = ["Dataset"]

# Specify the seed for reproducibility
seed = 42
np.random.seed(seed)

# Initialize lists to hold all data paths and their content
sequence_data_paths = []
sequence_total_data_paths = []

# Gather all file paths from the directories
for base_dir in base_dirs:
    # Loop through the numbered directories
    for i in range(1, 3503):
        number = str(i)
        dir_path = f"{base_dir}/{number}"
        if not os.path.isdir(dir_path):
            # If it doesn't exist, skip the rest of this loop iteration
            # print(f"Directory '{dir_path}' does not exist. Skipping...")
            continue
        
        sequence_total_path = os.path.join(base_dir, number, f'Sequence_total{number}.npy')
        sequence_total_data_paths.append(sequence_total_path)

# Shuffle indices for splitting
total_files = len(sequence_total_data_paths)
indices = np.arange(total_files)
np.random.shuffle(indices)

# Calculate the split index
split_index = int(total_files * 0.9)

# Split into training and validation indices
training_indices = indices[:split_index]
validation_indices = indices[split_index:]

# Initialize lists to hold training and validation data
training_total_data = []
validation_total_data = []

# Load training data
for idx in training_indices:
    training_total_data.append(np.load(sequence_total_data_paths[idx]))

# Load validation data
for idx in validation_indices:
    validation_total_data.append(np.load(sequence_total_data_paths[idx]))

# Convert lists to 2D numpy arrays
training_total_data = np.concatenate(training_total_data, axis=0)
validation_total_data = np.concatenate(validation_total_data, axis=0)

# Save the arrays
np.save('Training.npy', training_total_data)
np.save('Validation.npy', validation_total_data)

# Print the shapes of the training and validation data
print("Training total data shape:", training_total_data.shape)
print("Validation total data shape:", validation_total_data.shape)