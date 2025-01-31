import pickle

# Read the .pkl file
with open('NH3_cuda_sto-3g_R-HF_is_frozen.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract keys and values
keys = list(data.keys())
values = list(data.values())

# Save keys and values to separate text files
'''with open('keys_spin.txt', 'w') as file:
    for key in keys:
        file.write(f"{key}\n")'''

with open('NH3_ham.txt', 'w') as file:
    for value in values:
        file.write(f"{value}\n")
