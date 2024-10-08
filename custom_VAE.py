import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MIMICDATASET(Dataset):
    def __init__(self, x_t, x_s, y, train=None, transform=None):
        # Transform
        self.transform = transform
        self.train = train
        self.xt = x_t
        self.xs = x_s
        self.y = y

    def return_data(self):
        return self.xt, self.xs, self.y

    def __len__(self):
        return len(self.xt)

    def __getitem__(self, idx):
        sample = self.xt[idx]
        stat = self.xs[idx]
        sample_y = self.y[idx]
        return sample, stat, int(sample_y)


# CVAE 모델 정의
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim):
        super(CVAE, self).__init__()
        # 인코더
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 잠재 변수의 평균
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 잠재 변수의 로그 분산
        # 디코더
        self.fc3 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, c):
        # 입력 이미지와 조건 레이블을 결합
        x_cond = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(x_cond))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # 잠재 변수와 조건 레이블을 결합
        z_cond = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc3(z_cond))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


# 손실 함수 정의
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# 하이퍼파라미터 설정
batch_size = 512
learning_rate = 1e-5
num_epochs = 500

tasks = [
    'mortality_48h',
    'ARF_4h',
    'ARF_12h',
    'Shock_4h',
    'Shock_12h',
]
task = tasks[0]
dataset_name = "eICU"
# dataset_name = "MIMIC" # latent_dim=20, 1000 hidden_dim=50, 3000

s = np.load(f'FIDDLE_{dataset_name}/features/{task}/s.npz')
X = np.load(f'FIDDLE_{dataset_name}/features/{task}/X.npz')
s_feature_names = json.load(open(f'FIDDLE_{dataset_name}/features/{task}/s.feature_names.json', 'r'))
X_feature_names = json.load(open(f'FIDDLE_{dataset_name}/features/{task}/X.feature_names.json', 'r'))
df_pop = pd.read_csv(f'FIDDLE_{dataset_name}/population/{task}.csv')
x_s = torch.sparse_coo_tensor(torch.tensor(s['coords']), torch.tensor(s['data'])).to_dense().to(torch.float32)
x_t = torch.sparse_coo_tensor(torch.tensor(X['coords']), torch.tensor(X['data'])).to_dense().to(torch.float32)
x_t = x_t.mean(dim=1).to(torch.float32)
y = torch.tensor(df_pop["mortality_LABEL"].values).to(torch.float32).cuda()

idx = int(x_s.size()[0] * 0.8)
x_s_train = x_s[:idx]
x_t_train = x_t[:idx]
y_train = y[:idx]

x_s_test = x_s[idx:]
x_t_test = x_t[idx:]
y_test = y[idx:]

dataset_train = MIMICDATASET(x_t_train, x_s_train, y_train, train=True, transform=False)
dataset_test = MIMICDATASET(x_t_test, x_s_test, y_test, train=False, transform=False)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)

condition_dim = 2

# 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s_CVAE = CVAE(x_s.size(1), condition_dim, latent_dim=20, hidden_dim=50).to(device)
t_CVAE = CVAE(x_t.size(1), condition_dim, latent_dim=1000, hidden_dim=3000).to(device)
s_optimizer = torch.optim.Adam(s_CVAE.parameters(), lr=learning_rate)
t_optimizer = torch.optim.Adam(t_CVAE.parameters(), lr=learning_rate)

# 모델 훈련
print('Training...s_CVAE')
for epoch in range(num_epochs):
    s_CVAE.train()
    train_loss = 0
    for data_t, data_s, labels in train_loader:
        data_s = data_s.to(device)
        labels = labels.to(device)

        # 레이블을 원-핫 인코딩
        labels_onehot = F.one_hot(labels, num_classes=2).float().to(device)

        s_optimizer.zero_grad()
        s_recon_batch, s_mu, s_logvar = s_CVAE(data_s, labels_onehot)
        s_loss = loss_function(s_recon_batch, data_s, s_mu, s_logvar)
        s_loss.backward()

        train_loss += s_loss.item()
        s_optimizer.step()

    average_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}')

print('Training...t_CVAE')
for epoch in range(num_epochs):
    t_CVAE.train()
    train_loss = 0
    for data_t, data_s, labels in train_loader:
        data_t = data_t.to(device)
        labels = labels.to(device)

        # 레이블을 원-핫 인코딩
        labels_onehot = F.one_hot(labels, num_classes=2).float().to(device)

        t_optimizer.zero_grad()
        t_recon_batch, t_mu, t_logvar = t_CVAE(data_t.to(device), labels_onehot)
        t_loss = loss_function(t_recon_batch, data_t, t_mu, t_logvar)
        t_loss.backward()

        train_loss += t_loss.item()
        t_optimizer.step()

    average_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}')

torch.save(s_CVAE.state_dict(), f'saved_models_{dataset_name}/my_vae_stat.pth')
torch.save(t_CVAE.state_dict(), f'saved_models_{dataset_name}/my_vae_tmp.pth')


# 이미지 생성 및 시각화 함수
def generate_images(model, is_static, num_samples, label):
    model.eval()
    with torch.no_grad():
        # 조건 레이블 생성
        labels = torch.full((num_samples,), label, dtype=torch.long)
        labels_onehot = F.one_hot(labels, num_classes=condition_dim).float().to(device)
        # 잠재 변수 샘플링
        z = None
        if is_static:
            z = torch.randn(num_samples, 20).to(device)
        else:
            z = torch.randn(num_samples, 1000).to(device)
        # 이미지 생성
        generated = model.decode(z, labels_onehot).cpu()
        return generated


# 예시: 숫자 0에 대한 이미지 생성
s_syn_0 = generate_images(s_CVAE, is_static=True, num_samples=1000, label=0)
t_syn_0 = generate_images(t_CVAE, is_static=False, num_samples=1000, label=0)
s_syn_1 = generate_images(s_CVAE, is_static=True, num_samples=1000, label=1)
t_syn_1 = generate_images(t_CVAE, is_static=False, num_samples=1000, label=1)

s_syn = torch.cat([s_syn_0, s_syn_1], dim=0)
t_syn = torch.cat([t_syn_0, t_syn_1], dim=0)

np.save(f'Synthetic_{dataset_name}/my_vae_static.npy', s_syn)
np.save(f'Synthetic_{dataset_name}/my_vae_temporal.npy', t_syn)
