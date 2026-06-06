"""
CNN antrenat de la zero pentru clasificarea imaginilor in 5 clase.

Contine:
  - SignalDataset: clasa Dataset PyTorch care incarca + preproceseaza + augmenteaza
  - LineCNN: arhitectura (4 blocuri convolutionale + clasificator MLP)
  - train_fold: antrenare per fold cu early stopping
  - infereaza: predictie probabilitati + features, optional cu TTA (flip orizontal)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from preprocessing import preproceseaza
import config


class SignalDataset(Dataset):
    """Dataset PyTorch: incarca imagine, aplica preprocesare, optional augmenteaza.

    Augmentari (doar la train, niciuna nu schimba numarul de obiecte):
      - flip orizontal (inversare in timp; pante +/- raman tot linii)
      - translatie mica (+/- 3 pixeli) prin torch.roll
    """
    def __init__(self, df, img_dir, augment=False, test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = preproceseaza(Image.open(os.path.join(self.img_dir, r['id'])))
        t = torch.from_numpy(g).unsqueeze(0).float()    # (1, H, W)

        # daca dimensiunea difera de cea asteptata, redimensionam (rar necesar)
        if t.shape[1:] != (config.IMG_H, config.IMG_W):
            t = F.interpolate(t.unsqueeze(0), size=(config.IMG_H, config.IMG_W),
                              mode='bilinear', align_corners=False).squeeze(0)

        if self.augment:
            # flip orizontal cu probabilitate 0.5
            if np.random.rand() < 0.5:
                t = torch.flip(t, dims=[2])
            # translatie mica
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            t = torch.roll(t, shifts=(dy, dx), dims=(1, 2))

        if self.test:
            return t, r['id']
        return t, r['label'] - config.LABEL_OFFSET


class LineCNN(nn.Module):
    """CNN cu 4 blocuri convolutionale + clasificator MLP.

    Arhitectura: (32 -> 64 -> 128 -> 256 canale), fiecare bloc are 2x conv 3x3
    + BatchNorm + ReLU, urmat de MaxPool 2x2. Dropout2d 0.2 pe ultimul bloc.
    AdaptiveAvgPool la final -> vector de 256 trasaturi -> MLP cu dropout.
    """
    def __init__(self, n_out):
        super().__init__()
        def bloc_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            )
        self.b1 = bloc_conv(1,   32);  self.p1 = nn.MaxPool2d(2)
        self.b2 = bloc_conv(32,  64);  self.p2 = nn.MaxPool2d(2)
        self.b3 = bloc_conv(64,  128); self.p3 = nn.MaxPool2d(2)
        self.b4 = bloc_conv(128, 256)
        self.drop = nn.Dropout2d(0.2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(256, n_out),
        )

    def forward(self, x, return_feat=False):
        x = self.p1(self.b1(x))
        x = self.p2(self.b2(x))
        x = self.p3(self.b3(x))
        x = self.drop(self.b4(x))
        x = self.gap(x).flatten(1)   # (B, 256)
        if return_feat:
            return x   # pentru extragere de trasaturi (folosite de SVM, MLP)
        return self.classifier(x)


def infereaza(model, df, img_dir, device, tta=False, este_test=False):
    """Inferenta pe un subset.

    Returneaza (probabilitati, features, extra):
      - probabilitati: softmax (N, nr_clase)
      - features: vectorul de 256 dim. dinaintea clasificatorului
      - extra: etichetele (daca train/val) sau id-urile (daca test)
    Daca tta=True, mediem probabilitatile pe imaginea originala si pe cea flip orizontal.
    """
    ds = SignalDataset(df, img_dir, augment=False, test=este_test)
    ld = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    model.eval()
    P, Fe, extra = [], [], []
    with torch.no_grad():
        for im, ex in ld:
            im = im.to(device)
            p = torch.softmax(model(im), 1)
            if tta:
                p = (p + torch.softmax(model(torch.flip(im, [3])), 1)) / 2
            P.append(p.cpu().numpy())
            Fe.append(model(im, return_feat=True).cpu().numpy())
            if isinstance(ex, torch.Tensor):
                extra.extend(ex.numpy())
            else:
                extra.extend(ex)
    return np.concatenate(P), np.concatenate(Fe), extra


def train_fold(model_path, train_df, tr_idx, val_idx, class_weights, device, seed):
    """Antreneaza un model pe foldul (tr_idx, val_idx).

    Salveaza cel mai bun model (dupa val_acc) in model_path. Returneaza
    cea mai mare acuratete pe validare obtinuta. Early stopping cu min_epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds = SignalDataset(train_df.iloc[tr_idx], config.TRAIN_IMG_DIR, augment=True)
    va_ds = SignalDataset(train_df.iloc[val_idx], config.TRAIN_IMG_DIR, augment=False)
    tr = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                    num_workers=2, pin_memory=True)
    va = DataLoader(va_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                    num_workers=2, pin_memory=True)

    model = LineCNN(config.NR_CLASE).to(device)
    criteriu = nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=config.LABEL_SMOOTHING)
    optimizator = optim.Adam(model.parameters(), lr=config.LR,
                             weight_decay=config.WEIGHT_DECAY)
    programare = optim.lr_scheduler.OneCycleLR(
        optimizator, max_lr=config.MAX_LR,
        steps_per_epoch=len(tr), epochs=config.EPOCHS)

    best_acc, fara_progres = 0.0, 0
    for ep in range(config.EPOCHS):
        # antrenare
        model.train()
        for im, lb in tr:
            im, lb = im.to(device), lb.to(device)
            optimizator.zero_grad()
            loss = criteriu(model(im), lb)
            loss.backward()
            optimizator.step()
            programare.step()

        # validare
        model.eval()
        corecte = total = 0
        with torch.no_grad():
            for im, lb in va:
                im, lb = im.to(device), lb.to(device)
                corecte += (model(im).argmax(1) == lb).sum().item()
                total += im.size(0)
        acc = corecte / total

        if acc > best_acc:
            best_acc, fara_progres = acc, 0
            torch.save(model.state_dict(), model_path)
        else:
            fara_progres += 1

        # early stopping doar dupa min_epochs
        if ep + 1 >= config.MIN_EPOCHS and fara_progres >= config.PATIENCE:
            break

    return best_acc