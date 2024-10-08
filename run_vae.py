import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
import pandas as pd
from models.cvae import VariationalAutoencoder, vae_loss_fn
seed = 804


class MIMICDATASET(Dataset):
    def __init__(self, x_t,x_s, y, train=None, transform=None):
        # Transform
        self.transform = transform
        self.train = train
        self.xt = x_t
        self.xs = x_s
        self.y = y

    def return_data(self):
        return self.xt, self.xs, self.label

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, idx):
        sample = self.xt[idx]
        stat = self.xs[idx]
        sample_y = self.y[idx]
        return sample, stat, sample_y


def train_vae_stat(net, dataloader,  epochs=30):
    optim = torch.optim.Adam(net.parameters())
    for i in range(epochs):
        running_loss = 0
        for _, batch, y in dataloader:
            batch = batch.to(device)
            y = y.to(device)
            optim.zero_grad()
            x,mu, logvar, z = net(batch, y)
            loss = vae_loss_fn(batch, x, mu, logvar, numeric=False)
            loss.backward()
            optim.step()
            running_loss += loss.item()
        print(running_loss/512)
        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    torch.save(net.state_dict(), 'saved_models/vae_stat.pth')

def train_vae_tmp(net, dataloader,  epochs=30):
    optim = torch.optim.Adam(net.parameters())
    for i in range(epochs):
        running_loss = 0
        for batch, _,y in dataloader:
            y = y.to(device)
            batch = batch.to(device)
            optim.zero_grad()
            x,mu,logvar, z = net(batch, y)
            loss = vae_loss_fn(batch, x, mu, logvar, numeric=True)
            loss.backward()
            optim.step()
            running_loss += loss.item()
        print(running_loss/512)
        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    torch.save(net.state_dict(), 'saved_models/vae_tmp.pth')




if __name__ == "__main__":

    batch_size =  512
    device = torch.device("cuda")
    tasks = [
        'mortality_48h',
        'ARF_4h', 
        'ARF_12h',
        'Shock_4h',
        'Shock_12h',
    ]
    task = tasks[0]
    s = np.load('FIDDLE_mimic3/features/{}/s.npz'.format(task))
    X = np.load('FIDDLE_mimic3/features/{}/X.npz'.format(task))
    s_feature_names = json.load(open('FIDDLE_mimic3/features/{}/s.feature_names.json'.format(task), 'r'))
    X_feature_names = json.load(open('FIDDLE_mimic3/features/{}/X.feature_names.json'.format(task), 'r'))
    df_pop = pd.read_csv('FIDDLE_mimic3/population/{}.csv'.format(task))
    x_s = torch.sparse_coo_tensor(torch.tensor(s['coords']), torch.tensor(s['data'])).to_dense().to(torch.float32)
    x_t = torch.sparse_coo_tensor(torch.tensor(X['coords']), torch.tensor(X['data'])).to_dense().to(torch.float32)
    x_t = x_t.sum(dim=1).to(torch.float32)

    dataset_train_object = MIMICDATASET(x_t, x_s, torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32),\
                                         train=True, transform=False)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size, shuffle=True, \
                              num_workers=1, drop_last=False)


    tmp_samples, sta_samples, y = next(iter(train_loader))
    feature_dim_s = sta_samples.shape[1]
    feature_dim_t = tmp_samples.shape[1]

    print("VAE for static features")
    model = VariationalAutoencoder(feature_dim_s).to(device)
    train_vae_stat(model, train_loader,epochs=70)
    vae_sta_dict = torch.load('saved_models/vae_stat.pth', weights_only=True)
    vae_sta = VariationalAutoencoder(feature_dim_s).to(device)
    vae_sta.load_state_dict(vae_sta_dict)
    vae_sta.eval()

    print("VAE for temporal features")
    model2 = VariationalAutoencoder(feature_dim_t).to(device)
    train_vae_tmp(model2, train_loader,epochs=60)
    vae_tmp_dict = torch.load('saved_models/vae_tmp.pth', weights_only=True)
    vae_tmp = VariationalAutoencoder(feature_dim_t).to(device)
    vae_tmp.load_state_dict(vae_tmp_dict)
    vae_tmp.eval()
    
    with torch.no_grad():
        x_recon,mu,logvar, z = vae_tmp(x_t.cuda(),torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32).cuda())
        x_recon_s,mu,logvar, z = vae_sta(x_s.cuda(),\
                                         torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32).cuda())

        t_syn = x_recon.cpu().detach().numpy()
        s_syn = x_recon_s.cpu().detach().numpy()

        real_prob = np.mean(x_t.cpu().detach().numpy(), axis=0)
        fake_prob = np.mean(t_syn, axis=0)
        plt.scatter(real_prob, fake_prob)
        plt.title('Temporal')
        plt.xlabel('Real')
        plt.ylabel('Fake')
        plt.savefig('vae_scatter_plot_1.png')

        s_syn = np.round(1 / (1 + np.exp(-s_syn)))
        real_prob = np.mean(x_s.cpu().detach().numpy(), axis=0)
        fake_prob = np.mean(s_syn, axis=0)
        plt.scatter(real_prob, fake_prob)
        plt.title('Static')
        plt.xlabel('Real')
        plt.ylabel('Fake')
        plt.savefig('vae_scatter_plot_2.png')

        np.save("Synthetic_MIMIC/vae_static.npy", s_syn)
        np.save("Synthetic_MIMIC/vae_temporal.npy", t_syn)


