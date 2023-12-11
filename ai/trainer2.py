from AImover import AImover
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
        outputs = aimover(inputs)
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


df = pd.read_csv("dane_spowolnienie.csv")
data_size = df.shape[0]

batch_size = 8

train_size = round(data_size*0.8)
input_cols = ['start']
output_cols = ['end']
input_np_array = df[input_cols].to_numpy()
target_np_array = df[output_cols].to_numpy()
inputs_np = torch.tensor(input_np_array, dtype=torch.float)
targets_np = torch.tensor(target_np_array, dtype=torch.float)

ds = TensorDataset(inputs_np, targets_np)
generator = torch.Generator().manual_seed(42)
train_ds,val_ds = random_split(ds, [train_size,data_size-train_size],generator)
training_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_ds, batch_size = batch_size)

EPOCHS = 10
aimover = AImover()
#aimover.load_state_dict(torch.load("model2_best"))
optimizer = torch.optim.AdamW(aimover.parameters(),lr = 0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.01, end_factor=0.0000005, total_iters=EPOCHS)
loss_fn = torch.nn.MSELoss()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0



best_vloss = 1_000_000.
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    aimover.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    aimover.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = aimover(vinputs)
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
        torch.save(aimover.state_dict(), "model2_best")
    epoch_number += 1
    scheduler.step()
torch.save(aimover.state_dict(), "model2_last")
