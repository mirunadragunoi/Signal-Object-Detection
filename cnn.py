import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from preprocesare import preprocesare
import configurare

# antrenez un cnn de la zero ca sa clasific imaginile in 5 clase

class SignalDataset(Dataset):
    # incarc imaginile, aplic preprocesarea si optiunal augumentez
    # augumentare doar pe train!!!!

    def __init__(self, df, imagine_director, augment=False, test=False):
        self.df = df.reset_index(drop=True)
        self.imagine_director = imagine_director
        self.augment = augment
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        rand = self.df.iloc[index]
        matrice_imagine = preprocesare(Image.open(os.path.join(self.imagine_director, rand['id'])))
        tensor_imagine = torch.from_numpy(matrice_imagine).unsqueeze(0).float()

        # daca dimensiunea cumva difera de cea asteptata redimensionez
        if tensor_imagine.shape[1:] != (configurare.IMAGINE_INALTIME, configurare.IMAGINE_LUNGIME):
            tensor_imagine = F.interpolate(tensor_imagine.unsqueeze(0), size=(configurare.IMAGINE_INALTIME, configurare.IMAGINE_LUNGIME),
                              mode='bilinear', align_corners=False).squeeze(0)

        if self.augment:
            # flip orizontal cu probabilitate 0.5
            if np.random.rand() < 0.5:
                tensor_imagine = torch.flip(tensor_imagine, dims=[2])

            # translatie mica
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            tensor_imagine = torch.roll(tensor_imagine, shifts=(dy, dx), dims=(1, 2))

        if self.test:
            return tensor_imagine, rand['id']
        return tensor_imagine, rand['label'] - configurare.LABEL_OFFSET



class LineCNN(nn.Module):
    # cnn cu 4 blocuri convolutionale si clasificator MLP
    # 256 canale -->> fiecare bloc are 2x conv 3x3 + BatchNorm + ReLU, MaxPool 2x2 si Dropout2d pe ultimul bloc
    # AdaptiveAvgPool la final --->>> o sa am vector de 256 trasaturi --->>> MLP cu dropout

    def __init__(self, n_out):
        super().__init__()

        def bloc_convolutional(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            )

        self.b1 = bloc_convolutional(1,   32)
        self.p1 = nn.MaxPool2d(2)

        self.b2 = bloc_convolutional(32,  64)
        self.p2 = nn.MaxPool2d(2)

        self.b3 = bloc_convolutional(64,  128)
        self.p3 = nn.MaxPool2d(2)

        self.b4 = bloc_convolutional(128, 256)

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
        x = self.gap(x).flatten(1)

        if return_feat:
            return x

        return self.classifier(x)


def infereaza(model, df, imagine_director, device, tta=False, este_test=False):

    dataset = SignalDataset(df, imagine_director, augment=False, test=este_test)

    incarcator = DataLoader(dataset, batch_size=configurare.BATCH_DIMENSIUNE, shuffle=False, num_workers=2)

    model.eval()

    lista_probabilitati, lista_trasaturi, extra = [], [], []

    with torch.no_grad():
        for imagini_lot, meta_lot in incarcator:
            imagini_lot = imagini_lot.to(device)
            prob_lot = torch.softmax(model(imagini_lot), 1)

            if tta:
                prob_lot = (prob_lot + torch.softmax(model(torch.flip(imagini_lot, [3])), 1)) / 2

            lista_probabilitati.append(prob_lot.cpu().numpy())
            lista_trasaturi.append(model(imagini_lot, return_feat=True).cpu().numpy())

            if isinstance(meta_lot, torch.Tensor):
                extra.extend(meta_lot.numpy())
            else:
                extra.extend(meta_lot)

    return np.concatenate(lista_probabilitati), np.concatenate(lista_trasaturi), extra


def train_pe_fold(model_cale, train_df, train_index, valoare_index, ponderi_clase, device, seed):
    # antrenez un model pe fold ul train_index si valoare_index
    # salvez cel mai bun model dupa acuratete in model_cale

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_antrenare = SignalDataset(train_df.iloc[train_index], configurare.TRAIN_IMAGINI_DIRECTOR, augment=True)
    dataset_validare = SignalDataset(train_df.iloc[valoare_index], configurare.TRAIN_IMAGINI_DIRECTOR, augment=False)

    incarcator_antrenare = DataLoader(dataset_antrenare, batch_size=configurare.BATCH_DIMENSIUNE, shuffle=True, num_workers=2, pin_memory=True)
    incarcator_validare = DataLoader(dataset_validare, batch_size=configurare.BATCH_DIMENSIUNE, shuffle=False, num_workers=2, pin_memory=True)

    model = LineCNN(configurare.NR_CLASE).to(device)

    criteriu = nn.CrossEntropyLoss(weight=ponderi_clase, label_smoothing=configurare.NETEZIRE_ETICHETE)

    optimizator = optim.Adam(model.parameters(), lr=configurare.LR, weight_decay=configurare.REGULARIZARE_COEF)

    programare = optim.lr_scheduler.OneCycleLR(optimizator, max_lr=configurare.MAX_LR, steps_per_epoch=len(incarcator_antrenare),
                                               epochs=configurare.EPOCI)

    best_acuratete, fara_progres = 0.0, 0

    for epoca in range(configurare.EPOCI):

        # antrenare
        model.train()

        for imagini_lot, etichete_lot in incarcator_antrenare:
            imagini_lot, etichete_lot = imagini_lot.to(device), etichete_lot.to(device)
            optimizator.zero_grad()
            pierdere = criteriu(model(imagini_lot), etichete_lot)
            pierdere.backward()
            optimizator.step()
            programare.step()

        # validare
        model.eval()
        corecte = total = 0
        with torch.no_grad():
            for imagini_lot, etichete_lot in incarcator_validare:
                imagini_lot, etichete_lot = imagini_lot.to(device), etichete_lot.to(device)
                corecte += (model(imagini_lot).argmax(1) == etichete_lot).sum().item()
                total += imagini_lot.size(0)
        acuratete = corecte / total

        if acuratete > best_acuratete:
            best_acuratete, fara_progres = acuratete, 0
            torch.save(model.state_dict(), model_cale)
        else:
            fara_progres += 1

        # early stopping doar dupa mininul de epoci
        if epoca + 1 >= configurare.MIN_EPOCI and fara_progres >= configurare.RABDARE:
            break

    return best_acuratete

# functie pentru versiunea 10 a codului in care am implementat varianta de tta extinsa
def infereaza_tta_extinsa(model, df, imagine_director, device, tta=False, este_test=False):
    # mediez probabilitatile pe 5 versiuni ale fiecarei imagini
    # original, flip orizontal, translatie stg pe axa orizontala, translatie dr pe axa orizontala si flip orizonal
    # combinat cu translatie stg

    dataset = SignalDataset(df, imagine_director, augment=False, test=este_test)

    incarcator = DataLoader(dataset, batch_size=configurare.BATCH_DIMENSIUNE, shuffle=False, num_workers=2)

    model.eval()

    lista_probabilitati, lista_trasaturi, extra = [], [], []

    with torch.no_grad():
        for imagini_lot, meta_lot in incarcator:
            imagini_lot = imagini_lot.to(device)

            # trasaturile doar pt imaginea originala ca sa fie consistena cu oof
            lista_trasaturi.append(model(imagini_lot, return_feat=True).cpu().numpy())

            # cele 5 versiuni pt predictie
            imagine_flip = torch.flip(imagini_lot, dims=[3])
            imagine_stanga = torch.roll(imagini_lot, shifts=-2, dims=3)
            imagine_dreapta = torch.roll(imagini_lot, shifts=2, dims=3)
            imagine_flip_stanga = torch.roll(imagine_flip, shifts=-2, dims=3)

            prob_lot = (torch.softmax(model(imagini_lot), 1) +
                        torch.softmax(model(imagine_flip), 1) +
                        torch.softmax(model(imagine_stanga), 1) +
                        torch.softmax(model(imagine_dreapta), 1) +
                        torch.softmax(model(imagine_flip_stanga), 1)
                        ) / 5

            lista_probabilitati.append(prob_lot.cpu().numpy())

            if isinstance(meta_lot, torch.Tensor):
                extra.extend(meta_lot.numpy())
            else:
                extra.extend(meta_lot)

    return np.concatenate(lista_probabilitati), np.concatenate(lista_trasaturi), extra