import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = "cuda"

def show(tensor:torch.Tensor, ch=1, size=(512, 512), num=25):
    data = tensor.detach().cpu().view(-1, ch, *size)
    grid = make_grid(data[:num], nrow=5).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()

def gen_noise(batch_size, z_dim):
    return torch.randn(batch_size,z_dim).to(device)

def train_single_epoch(model:nn.Module,
                       data_loader:DataLoader,
                       loss_function:nn.Module,
                       optimizer:torch.optim.Optimizer,
                       compute_device:torch.device,
                       accuracy_fn = False):
    train_loss,train_acc = 0,0

    model.train()

    for batch , (X,y) in enumerate(data_loader):
        X,y = X.to(compute_device),y.to(compute_device)
        y = convert_labels_to_tensors(y,2)
        y_logits = model(X)
        y_pred = torch.sigmoid(y_logits)
        loss = loss_function(y_logits,y)
        train_loss += loss
        if(accuracy_fn): 
            train_acc += accuracy_fn(y_true=y.squeeze().argmax(dim=1),y_pred=y_pred.squeeze().argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_acc /= len(data_loader)
    train_loss /= len(data_loader)
    if(accuracy_fn):            
        print(f"Training_Loss: {train_loss} | Training_Acc: {train_acc} ")
            

    return train_loss,train_acc

def convert_labels_to_tensors(y:torch.Tensor,number_of_labels):
    y_labels = torch.zeros((len(y),number_of_labels)).to(y.device)
    for i in range(len(y)):
        y_labels[i][y[i]] = 1
    return y_labels

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def disc_loss(loss_func, gen, disc, batch_size, z_dim, real):

    noise = gen_noise(batch_size, z_dim)
    fake = gen(noise)
    disc_fake = disc(fake.detach())
    disc_fake_target = torch.zeros_like(disc_fake)
    disc_fake_loss = loss_func(disc_fake, disc_fake_target)

    disc_real = disc(real)
    disc_real_target = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, disc_real_target)

    return (disc_fake_loss + disc_real_loss) / 2

def gen_loss(loss_func, gen, disc, batch_size, z_dim):
    noise = gen_noise(batch_size, z_dim)
    fake = gen(noise)
    pred = disc(fake)
    target = torch.ones_like(pred)
    return loss_func(pred, target)