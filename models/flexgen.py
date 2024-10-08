import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch import nn, optim
from tqdm import tqdm
from models.cvae import vae_loss_fn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda'
class flexgen(nn.Module):
    def __init__(self, tvae, svae, ldm, trainloader, dataset_name, epochs=10):
        super(flexgen, self).__init__()
        self.tvae = tvae
        self.svae = svae
        self.diff = ldm
        self.dopt = optim.Adam(self.diff.parameters(), lr=1e-4)
        self.train = trainloader
        self.n_epoch = epochs
        self.tvae.eval()
        self.svae.eval()
        self.xt = trainloader.dataset.xt
        self.xs = trainloader.dataset.xs
        self.m_path = f'saved_models_{dataset_name}/flexgenDiff.pth'
        self.s_path = f'Synthetic_{dataset_name}/flexgen_static.npy'
        self.t_path = f'Synthetic_{dataset_name}/flexgen_temporal.npy'


    def train_epoch(self):
        self.diff.train()
        loss_history = []
        epoch_tqdm = tqdm(range(self.n_epoch))
        for ep in epoch_tqdm:
            # print(f'epoch {ep}')
            # linear lrate decay
            self.dopt.param_groups[0]['lr'] = 1e-4*(1-ep/self.n_epoch)

            # pbar = tqdm(self.train)
            for xt, xs, c in self.train:
                self.dopt.zero_grad()
                xt = xt.to(device)
                xs = xs.to(device)
                c = c.to(device)
                # zt, _ =  self.tvae.encode(xt)
                # zs, _ =  self.svae.encode(xs)
                # z = torch.concat([zt, zs],dim=1)
                c2 = torch.nn.functional.one_hot(c, num_classes=2).float().to(device)

                ms, ss = self.svae.encode(xs, c2) # 수정
                zs = self.svae.reparameterize(ms, ss)
                mt, st = self.tvae.encode(xt, c2) # 수정
                zt = self.tvae.reparameterize(mt, st)
                z = torch.concat([zt, zs],dim=1)

                loss_d = self.diff(z, c)
                loss_d.backward()
                # z_gen, _ = self.diff.sample(xt.shape[0], (256,), device, label=c, guide_w=0.5)
                # z_t, z_s = torch.chunk(z_gen, 2, 1)
                # xt_gen = self.tvae.decode(z_t)
                # xs_gen = self.svae.decode(z_s)

                self.dopt.step()
            epoch_tqdm.set_description(f"loss: {loss_d.item():.4f}")
            loss_history.append(loss_d.item())

        torch.save(self.diff.state_dict(), self.m_path)
        plt.plot(loss_history)
        plt.title('Training Loss')
        plt.show()


    def generate(self, num_sample, label, eval=True):
        diff_dict = torch.load(self.m_path, weights_only=True)
        self.diff.load_state_dict(diff_dict)
        diff = self.diff
        diff.eval()
        with torch.no_grad():
            z_gen, _ = diff.sample(num_sample, (1020,), device, label=[label], guide_w=0.5)
            # z_t, z_s = torch.chunk(z_gen, 2, 1)
            z_t = z_gen[:, :1000]
            z_s = z_gen[:, 1000:]

            labels = torch.tensor(label).to(device).unsqueeze(0).repeat(num_sample)
            labels = torch.nn.functional.one_hot(labels, num_classes=2).float().to(device)
            print(z_t.shape, labels.shape)
            xt_gen = self.tvae.decode(z_t, labels)
            xs_gen = self.svae.decode(z_s, labels)


        t_syn = xt_gen.cpu().detach().numpy()  # synthetic temporal records
        s_syn = xs_gen.cpu().detach().numpy()  # synthetic static records

        t_real_prob = np.mean(self.xt.cpu().detach().numpy(), axis=0)
        t_fake_prob = np.mean(t_syn, axis=0)
        plt.scatter(t_real_prob, t_fake_prob)
        plt.title(f'Temporal_{label}')
        plt.xlabel('Real')
        plt.ylabel('Fake')
        plt.show()

        s_real_prob = np.mean(self.xs.cpu().detach().numpy(), axis=0)
        s_fake_prob = np.mean(s_syn, axis=0)
        plt.scatter(s_real_prob, s_fake_prob)
        plt.title(f'Static_{label}')
        plt.xlabel('Real')
        plt.ylabel('Fake')
        plt.show()

        return s_syn, t_syn