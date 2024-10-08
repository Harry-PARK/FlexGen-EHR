import torch
import torch.utils.data
from Cython.Shadow import numeric
from matplotlib.font_manager import weight_dict
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
from sklearn.model_selection import train_test_split
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


def validate_vae(net, dataloader):
    net.eval()  # 모델을 평가 모드로 전환
    validation_loss = 0
    with torch.no_grad():  # 평가 시에는 gradient 계산을 하지 않음
        for batch, _, y in dataloader:
            y = y.to(device)
            batch = batch.to(device)
            x, mu, logvar, z = net(batch, y)
            loss = vae_loss_fn(batch, x, mu, logvar, numeric=True)
            validation_loss += loss.item()

    average_validation_loss = validation_loss / len(dataloader.dataset)
    print(f'Validation Loss: {average_validation_loss}')
    return average_validation_loss


def train_vae(net, train_dataloader, dataset_name, static, epochs=30,):
    train_loss_history = []
    val_loss_history = []
    numeric = None
    model_name = None
    optim = torch.optim.Adam(net.parameters())

    for i in range(epochs):
        net.train()
        loss = 0
        running_loss = 0
        for batch_tmp, batch_sta, y in train_dataloader:
            if static:
                batch = batch_sta
                numeric = False
                model_name = 'vae_stat'
            else:
                batch = batch_tmp
                numeric = True
                model_name = 'vae_tmp'
            batch = batch.to(device)
            y = y.to(device)

            optim.zero_grad()
            x, mu, logvar, z = net(batch, y)
            loss = vae_loss_fn(batch, x, mu, logvar, numeric=numeric)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        print(running_loss / 512)
        train_loss_history.append(loss.item()/ 512)
        # validation_loss = validate_vae(net, test_dataloader)
        # val_loss_history.append(validation_loss)

        # evaluate(validation_losses, net, test_dataloader, vae=True, title=f'VAE - Epoch {i}')
    torch.save(net.state_dict(), f'saved_models_{dataset_name}/{model_name}.pth')
    return train_loss_history
    # return train_loss_history, val_loss_history





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
    # dataset_name = "eICU"
    dataset_name = "MIMIC"

    s = np.load(f'FIDDLE_{dataset_name}/features/{task}/s.npz')
    X = np.load(f'FIDDLE_{dataset_name}/features/{task}/X.npz')
    s_feature_names = json.load(open(f'FIDDLE_{dataset_name}/features/{task}/s.feature_names.json', 'r'))
    X_feature_names = json.load(open(f'FIDDLE_{dataset_name}/features/{task}/X.feature_names.json', 'r'))
    df_pop = pd.read_csv(f'FIDDLE_{dataset_name}/population/{task}.csv')
    x_s = torch.sparse_coo_tensor(torch.tensor(s['coords']), torch.tensor(s['data'])).to_dense().to(torch.float32)
    x_t = torch.sparse_coo_tensor(torch.tensor(X['coords']), torch.tensor(X['data'])).to_dense().to(torch.float32)
    x_t = x_t.sum(dim=1).to(torch.float32)
    y = torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32).cuda()


    # train_idx = int(x_s.size()[0]*0.8)
    #
    # x_s_train = x_s[:train_idx]
    # x_t_train = x_t[:train_idx]
    # y_train = y[:train_idx]
    #
    # x_s_test = x_s[train_idx:]
    # x_t_test = x_t[train_idx:]
    # y_test = y[train_idx:]
    #
    # print(x_s_train.size(), x_t_train.size(), y_train.size())
    # print(x_s_test.size(), x_t_test.size(), y_test.size())
    #
    # # 데이터셋 객체 생성
    # train_dataset = MIMICDATASET(x_t_train, x_s_train, y_train, train=True, transform=False)
    # test_dataset = MIMICDATASET(x_t_test, x_s_test, y_test, train=False, transform=False)
    # # 데이터 로더 생성
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    dataset_train_object = MIMICDATASET(x_t, x_s, torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32),\
                                         train=True, transform=False)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size, shuffle=True, \
                              num_workers=1, drop_last=False)



    tmp_samples, sta_samples, y = next(iter(train_loader))
    feature_dim_s = sta_samples.shape[1]
    feature_dim_t = tmp_samples.shape[1]


    print("VAE for static features")
    model = VariationalAutoencoder(feature_dim_s).to(device)
    vae_sta_tl = train_vae(model, train_loader, dataset_name, static=True, epochs=500)
    vae_sta = VariationalAutoencoder(feature_dim_s).to(device)
    vae_sta_dict = torch.load(f'saved_models_{dataset_name}/vae_stat.pth', weights_only=True)
    vae_sta.load_state_dict(vae_sta_dict)
    vae_sta.eval()

    print("VAE for temporal features")
    model2 = VariationalAutoencoder(feature_dim_t).to(device)
    vae_tmp_tl = train_vae(model2, train_loader, dataset_name, static=False, epochs=500)
    vae_tmp = VariationalAutoencoder(feature_dim_t).to(device)
    vae_tmp_dict = torch.load(f'saved_models_{dataset_name}/vae_tmp.pth', weights_only=True)
    vae_tmp.load_state_dict(vae_tmp_dict)
    vae_tmp.eval()


    # print vae_sta_loss
    plt.plot(vae_sta_tl)
    # plt.plot(vae_sta_vl)
    plt.title('VAE Static Loss')
    plt.legend(['Train_loss', 'Validation_loss'])
    plt.show()

    plt.plot(vae_tmp_tl)
    # plt.plot(vae_tmp_vl)
    plt.title('VAE Temporal Loss')
    plt.legend(['Train_loss', 'Validation_loss'])
    plt.show()


    with torch.no_grad():
        y = torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32).cuda()
        x_recon_t,mu,logvar, z = vae_tmp(x_t.cuda(), 0)
        x_recon_s,mu,logvar, z = vae_sta(x_s.cuda(), 1)

        t_syn = x_recon_t.cpu().detach().numpy()
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

        np.save(f"Synthetic_{dataset_name}/vae_static.npy", s_syn)
        np.save(f"Synthetic_{dataset_name}/vae_temporal.npy", t_syn)


