from AIcollider import AIcolider
import pandas as pd
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
from torch.utils.data import random_split

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = aicollider(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

batch_size = 32
df = pd.read_csv("data.csv")
data_size = df.shape[0]
train_size = round(data_size*0.8)
input_cols = ['wektor_X','wektor_Y','x_start1','y_start1','x_start2','y_start2']
output_cols = ['x_koniec1','y_koniec1','x_koniec2','y_koniec2']
input_np_array = df[input_cols].to_numpy()
target_np_array = df[output_cols].to_numpy()
inputs_np = torch.tensor(input_np_array, dtype=torch.float)
targets_np = torch.tensor(target_np_array, dtype=torch.float)

#normalize the data
"""dt_mean_input = torch.mean(inputs_np,0)
dt_mean_output = torch.mean(targets_np,0)
dt_std_input = torch.std(inputs_np,0)
dt_std_output = torch.std(targets_np,0)

print(dt_mean_input)
print(dt_mean_output)
print(dt_std_input)
print(dt_std_output)

inputs_normalized = (inputs_np - dt_mean_input)/dt_std_input
outputs_normalized = (targets_np - dt_mean_output)/dt_std_output"""


ds = TensorDataset(inputs_np, targets_np)
generator = torch.Generator().manual_seed(42)
train_ds,val_ds = random_split(ds, [train_size,data_size-train_size],generator)
training_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_ds, batch_size = batch_size)

EPOCHS = 100
aicollider = AIcolider()
aicollider.load_state_dict(torch.load("model_best"))
optimizer = torch.optim.AdamW(aicollider.parameters(),lr = 0.0000001)
loss_fn = torch.nn.MSELoss()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0



best_vloss = 1_000_000.
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    aicollider.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    aicollider.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = aicollider(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}\n'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(aicollider.state_dict(), "model_best")
    epoch_number += 1
torch.save(aicollider.state_dict(), "model_last")
