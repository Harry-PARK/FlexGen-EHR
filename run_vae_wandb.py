import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import wandb
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
    def __init__(self, x_t, x_s, y, train=None, transform=None):
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


def train_vae(net, train_dataloader, dataset_name, static, epochs, config):
    train_loss_history = []
    numeric = None
    model_name = None
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(epochs):  # epochs는 config에서 가져오는 것이 아닌 인자로 전달됨
        net.train()
        running_loss = 0.0
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

            optimizer.zero_grad()
            x, mu, logvar, z = net(batch, y)
            loss = vae_loss_fn(batch, x, mu, logvar, numeric=numeric)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / 512
        train_loss_history.append(avg_loss)

        # WandB에 학습 손실 기록
        wandb.log({"epoch": epoch, "train_loss": avg_loss})

        print(f"Epoch {epoch}: Loss = {avg_loss}")

    torch.save(net.state_dict(), f'saved_models_{dataset_name}/wandb/{model_name}_{config.epochs}_{config.hidden_size}_{config.out}.pth')
    return train_loss_history


if __name__ == "__main__":
    # 데이터 관련 설정
    batch_size = 512
    device = torch.device("cuda")
    tasks = [
        'mortality_48h',
        'ARF_4h',
        'ARF_12h',
        'Shock_4h',
        'Shock_12h',
    ]
    task = tasks[0]
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

    dataset_train_object = MIMICDATASET(x_t, x_s, torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32), \
                                        train=True, transform=False)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size, shuffle=True, \
                              num_workers=1, drop_last=False)

    tmp_samples, sta_samples, y = next(iter(train_loader))
    feature_dim_s = sta_samples.shape[1]
    feature_dim_t = tmp_samples.shape[1]

    # WandB 스위프 설정
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'epochs': {
                'values': [100, 200, 300, 400, 500]
            },
            'hidden_size': {
                'values': [128, 256]
            },
            'out': {
                'values': [64, 128]
            }
        }
    }

    # 스위프 생성
    sweep_id = wandb.sweep(sweep_config, project="vae-hyperparameter-tuning")

    # 스위프 실행 함수
    def sweep_train():
        # WandB 스위프 설정에 따른 학습 실행
        with wandb.init() as run:
            config = run.config

            # VAE for static features
            model = VariationalAutoencoder(feature_dim_s, hidden=config.hidden_size, out=config.out).to(device)
            vae_sta_tl = train_vae(model, train_loader, dataset_name, static=True, epochs=config.epochs, config=config)

            # VAE for temporal features
            model2 = VariationalAutoencoder(feature_dim_t, hidden=config.hidden_size, out=config.out).to(device)
            vae_tmp_tl = train_vae(model2, train_loader, dataset_name, static=False, epochs=config.epochs, config=config)

            # VAE static loss plot
            # plt.plot(vae_sta_tl)
            # plt.title('VAE Static Loss')
            # plt.legend(['Train_loss'])
            # plt.show()

            # VAE temporal loss plot
            # plt.plot(vae_tmp_tl)
            # plt.title('VAE Temporal Loss')
            # plt.legend(['Train_loss'])
            # plt.show()

            # Synthetic data 저장
            with torch.no_grad():
                y = torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32).cuda()
                x_recon_t, mu, logvar, z = model2(x_t.cuda(), y)
                x_recon_s, mu, logvar, z = model(x_s.cuda(), y)

                t_syn = x_recon_t.cpu().detach().numpy()
                s_syn = x_recon_s.cpu().detach().numpy()

                real_prob = np.mean(x_t.cpu().detach().numpy(), axis=0)
                fake_prob = np.mean(t_syn, axis=0)
                plt.scatter(real_prob, fake_prob)


                s_syn = np.round(1 / (1 + np.exp(-s_syn)))
                real_prob = np.mean(x_s.cpu().detach().numpy(), axis=0)
                fake_prob = np.mean(s_syn, axis=0)


                np.save(f"Synthetic_{dataset_name}/wandb/vae_static_{config.epochs}_{config.hidden_size}_{config.out}.npy", s_syn)
                np.save(f"Synthetic_{dataset_name}/wandb/vae_temporal_{config.epochs}_{config.hidden_size}_{config.out}.npy", t_syn)

    # 스위프 실행
    wandb.agent(sweep_id, function=sweep_train)
