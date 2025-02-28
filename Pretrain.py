import torch
from torch.nn import functional as F
import numpy as np
import csv
import os
from Models.GPT import GPTLanguageModel

# hyperparameters
batch_size = 64
block_size = 1024
max_iters = 100000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 1000
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

filename = 'Pretrain'
Trainingdata = 'Training.npy'
Validationdata = 'Validation.npy'

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

train_data = np.load(Trainingdata)
train_data = torch.tensor(encode(train_data.flatten()), dtype=torch.long).view(train_data.shape)
val_data = np.load(Validationdata)
val_data = torch.tensor(encode(val_data.flatten()), dtype=torch.long).view(val_data.shape)
print(train_data.shape)
print(val_data.shape)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i,:-1] for i in ix])
    y = torch.stack([data[i,1:] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        filtered_losses = torch.zeros(eval_iters)  # To store the new filtered loss
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

            # Flatten logits and targets
            BandT, C = logits.shape
            targets = Y.view(BandT)

            # Exclude entries where targets == stoi["TRUNCATE"]
            truncate_mask = (targets != stoi["TRUNCATE"])  # Mask for valid targets
            logits_filtered = logits[truncate_mask]
            targets_filtered = targets[truncate_mask]

            # Calculate the filtered loss
            filtered_loss = F.cross_entropy(logits_filtered, targets_filtered)
            filtered_losses[k] = filtered_loss.item()
        
        out[split] = losses.mean()
        out[f'{split}_filtered_loss'] = filtered_losses.mean()

    model.train()
    return out

model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
val_loss = float('inf')

# Open a CSV file to write the losses
csv_file = open(filename+'.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Step', 'Train Loss', 'Train Filtered Loss', 'Validation Loss', 'Validation Filtered Loss'])

for iter in range(max_iters):
    torch.cuda.empty_cache()

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, train filtered loss {losses['train_filtered_loss']:.4f}, val loss {losses['val']:.4f}, val filtered loss {losses['val_filtered_loss']:.4f}")
        csv_writer.writerow([iter, losses['train'].item(), losses['train_filtered_loss'].item(), losses['val'].item(), losses['val_filtered_loss'].item()]) # Write losses to CSV
        if losses['val'] < val_loss:
            savemodel_name = filename + '.pth'
            torch.save(model.state_dict(), savemodel_name)
            val_loss = losses['val']

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()