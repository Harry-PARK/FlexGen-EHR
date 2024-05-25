from models.cvae import VariationalAutoencoder, vae_loss_fn
from models.ddpm import DDPM, ContextUnet
from models.meddiff import MedDiff
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
import pandas as pd
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
from tqdm import tqdm
import umap
from ot import Pamona

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

if __name__ == '__main__':

    batch_size =  512
    device = torch.device("cuda")
    tasks = [
        'mortality_48h',
        'ARF_4h', 
        'ARF_12h',
        'Shock_4h',
        'Shock_12h',
    ]
    task = tasks[1]
    s = np.load('FIDDLE_eicu/features/{}/s.npz'.format(task))
    X = np.load('FIDDLE_eicu/features/{}/X.npz'.format(task))
    s_feature_names = json.load(open('FIDDLE_eicu/features/{}/s.feature_names.json'.format(task), 'r'))
    X_feature_names = json.load(open('FIDDLE_eicu/features/{}/X.feature_names.json'.format(task), 'r'))
    df_pop = pd.read_csv('FIDDLE_eicu/population/{}.csv'.format(task))
    x_s = torch.sparse_coo_tensor(torch.tensor(s['coords']), torch.tensor(s['data'])).to_dense().to(torch.float32)
    x_t = torch.sparse_coo_tensor(torch.tensor(X['coords']), torch.tensor(X['data'])).to_dense().to(torch.float32)
    x_t = x_t.sum(dim=1).to(torch.float32)
    y = torch.tensor(df_pop.ARF_LABEL.values).to(torch.float32)
    dataset_train_object = MIMICDATASET(x_t, x_s, y,\
                                         train=True, transform=False)
    train_loader = DataLoader(dataset_train_object, batch_size=batch_size, shuffle=True, \
                              num_workers=1, drop_last=False)
    tmp_samples, sta_samples, yb = next(iter(train_loader))
    feature_dim_s = sta_samples.shape[1]
    feature_dim_t = tmp_samples.shape[1]
    svae = VariationalAutoencoder(feature_dim_s).to(device)
    tvae = VariationalAutoencoder(feature_dim_t).to(device)
    tvae = torch.load('saved_models/vae_tmp.pt')
    svae = torch.load('saved_models/vae_stat.pt')
    svae.eval()
    tvae.eval()

    _,_,_, zt = tvae(x_t.to(device), y.to(device))
    _,_,_, zs = svae(x_s.to(device), y.to(device))
    # ms, ss =svae.encode(x_s.to(device))
    # zs = svae.reparameterize(ms, ss)
    # mt, st = tvae.encode(x_t.to(device))
    # zt = tvae.reparameterize(mt, st)
    idx_t = np.random.permutation(np.arange(zt.shape[0]))[0: int(0.8 * zs.shape[0])]
    idx_s = np.random.permutation(np.arange(zs.shape[0]))[0: int(0.8 * zs.shape[0])]
    zs1 = zs[idx_s]
    zt1 = zt[idx_t]
    zs = zs1.detach().cpu().numpy().astype(np.float32)[0:5000, :]
    zt = zt1.detach().cpu().numpy().astype(np.float32)[0:5000, :]
    data = [zs, zt]
    y = df_pop.ARF_LABEL.values[0:5000]
    y2 = df_pop.ARF_LABEL.values[0:5000]
    datatype = [y.astype(np.int32), y2.astype(np.int32)]
    M = []
    n_datasets = len(data)
    for k in range(n_datasets-1):
        M.append(np.ones((len(data[k]), len(data[-1]))))
        for i in range(len(data[k])):
            for j in range(len(data[-1])):
                if datatype[k][i] == datatype[-1][j]:
                    M[k][i][j] = 0.5
    Pa = Pamona(n_shared=[500], M=M, n_neighbors=5)
    integrated_data, T = Pa.run_Pamona(data)
    Pa.Visualize(data, integrated_data, datatype=datatype, mode='UMAP')
    