import torch
from torch.nn import functional as F
import numpy as np
import csv
import os
from Models.GPT import GPTLanguageModel

# hyperparameters
block_size = 1024 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

filename = 'Inference'

torch.manual_seed(1337)

# Tokenizer
# Define the device base names
nm_np_bases = ["{}_D", "{}_G", "{}_S", "{}_B"]
npn_pnp_bases = ["{}_C", "{}_B", "{}_E"]
c_r_l_i_bases = ["{}_P", "{}_N"]
xor_bases = ["{}_A", "{}_B", "{}_VDD", "{}_VSS", "{}_Y"]
pfd_bases = ["{}_A", "{}_B", "{}_QA", "{}_QB", "{}_VDD", "{}_VSS"]
inverter_bases = ["{}_A", "{}_Q", "{}_VDD", "{}_VSS"]
transmission_gate_bases = ["{}_A", "{}_B", "{}_C", "{}_VDD", "{}_VSS"]

# Initialize the list of NM, PM, C, R, L, I, VIN, VB, VOUT devices, and additional entries
devices = []
for prefix in ["NM", "PM"]:
    for i in range(1, 35):
        devices.append(f"{prefix}{i}")
        for base in nm_np_bases:
            devices.append(base.format(f"{prefix}{i}"))

for prefix in ["NPN", "PNP"]:
    for i in range(1, 27):
        devices.append(f"{prefix}{i}")
        for base in npn_pnp_bases:
            devices.append(base.format(f"{prefix}{i}"))

for i in range(1, 28):
    devices.append(f"R{i}")
    for base in c_r_l_i_bases:
        devices.append(base.format(f"R{i}"))

for i in range(1, 16):
    devices.append(f"C{i}")
    for base in c_r_l_i_bases:
        devices.append(base.format(f"C{i}"))

for i in range(1, 24):
    devices.append(f"L{i}")
    for base in c_r_l_i_bases:
        devices.append(base.format(f"L{i}"))

for i in range(1, 8):
    devices.append(f"DIO{i}")
    for base in c_r_l_i_bases:
        devices.append(base.format(f"DIO{i}"))

for i in range(1, 2):
    devices.append(f"XOR{i}")
    for base in xor_bases:
        devices.append(base.format(f"XOR{i}"))

for i in range(1, 2):
    devices.append(f"PFD{i}")
    for base in pfd_bases:
        devices.append(base.format(f"PFD{i}"))

for i in range(1, 11):
    devices.append(f"INVERTER{i}")
    for base in inverter_bases:
        devices.append(base.format(f"INVERTER{i}"))

for i in range(1, 13):
    devices.append(f"TRANSMISSION_GATE{i}")
    for base in transmission_gate_bases:
        devices.append(base.format(f"TRANSMISSION_GATE{i}"))

for i in range(1, 11):
    devices.append(f"VIN{i}")

for i in range(1, 3):
    devices.append(f"IIN{i}")

for i in range(1, 7):
    devices.append(f"VOUT{i}")

for i in range(1, 5):
    devices.append(f"IOUT{i}")

for i in range(1, 11):
    devices.append(f"VB{i}")

for i in range(1, 7):
    devices.append(f"IB{i}")

for i in range(1, 21):
    devices.append(f"VCONT{i}")

for i in range(1, 9):
    devices.append(f"VCLK{i}")

for i in range(1, 3):
    devices.append(f"VCM{i}")

for i in range(1, 3):
    devices.append(f"VREF{i}")

for i in range(1, 3):
    devices.append(f"IREF{i}")

for i in range(1, 3):
    devices.append(f"VRF{i}")

for i in range(1, 5):
    devices.append(f"VLO{i}")

for i in range(1, 3):
    devices.append(f"VIF{i}")

for i in range(1, 5):
    devices.append(f"VBB{i}")

for i in range(1, 3):
    devices.append(f"LOGICA{i}")

for i in range(1, 3):
    devices.append(f"LOGICB{i}")

for i in range(1, 3):
    devices.append(f"LOGICD{i}")

for i in range(1, 3):
    devices.append(f"LOGICF{i}")

for i in range(1, 3):
    devices.append(f"LOGICG{i}")

for i in range(1, 3):
    devices.append(f"LOGICQ{i}")

for i in range(1, 2):
    devices.append(f"LOGICQA{i}")

for i in range(1, 2):
    devices.append(f"LOGICQB{i}")

for i in range(1, 3):
    devices.append(f"VLATCH{i}")

for i in range(1, 2):
    devices.append(f"VHOLD{i}")

for i in range(1, 3):
    devices.append(f"VTRACK{i}")

# Adding the additional entries
additional_entries = ["VDD", "VSS", "TRUNCATE"]
devices.extend(additional_entries)

# Create a mapping from device names to integers and vice versa
stoi = { device: i for i, device in enumerate(devices) }
itos = { i:device for i,device in enumerate(devices) }
vocab_size = len(devices)

# Print the results
print("Devices in order:", devices)
print("Vocabulary size:", len(devices))
print("Device to index mapping:", stoi)
print("Index to device mapping:", itos)

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: '->'.join([itos[i] for i in l]) + '->'

model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

savemodel_name = 'Pretrain.pth'
model.load_state_dict(torch.load(savemodel_name),strict=False)
run = 1000
os.makedirs(filename, exist_ok=True)
for i in range(run):
    context = torch.full((1, 1), 1003, dtype=torch.long, device=device)
    save_dir = filename + '/run'+str(i)+'.txt'
    sequence = m.generate(context, max_new_tokens=1024)[0].tolist()
    open(save_dir, 'w').write(decode(sequence))