import os
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import sobel

from preprocesare import preprocesare

def extrag_trasaturi_manuale(imagine):
    # extrag cele 52 trasaturi din imaginea preprocesata
    trasaturi = []

    # proiectiile pe axe
    proiectie_coloana = imagine.sum(axis=0)
    proiectie_rand = imagine.sum(axis=1)

    trasaturi.extend([proiectie_coloana.mean(), proiectie_coloana.std(), proiectie_coloana.max(), proiectie_coloana.min()])
    trasaturi.extend([proiectie_rand.mean(), proiectie_rand.std(), proiectie_rand.max(), proiectie_rand.min()])

    # peak uri pe proiectii
    for proiectie in [proiectie_coloana, proiectie_rand]:
        # normalizez in 0 1 ca sa am praguri consistente intre imagini
        pro = (proiectie - proiectie.min()) / (proiectie.max() - proiectie.min() + 1e-6)

        peaks, _ = find_peaks(pro, prominence=0.1, distance=2)

        if len(peaks) > 0:
            inalt = pro[peaks]

            trasaturi.extend([
                len(peaks),
                inalt.mean(),
                inalt.std() if len(peaks) > 1 else 0,
                inalt.max(),
                np.diff(peaks).mean() if len(peaks) > 1 else 0,
            ])
        else:
            trasaturi.extend([0, 0, 0, 0, 0])

        # peak uri la proeminenta mai mare inseamna ca s linii puternice si la mai mica am linii slabe
        trasaturi.append(len(find_peaks(pro, prominence=0.25, distance=2)[0]))
        trasaturi.append(len(find_peaks(pro, prominence=0.05, distance=2)[0]))

    # filtre sobel directionale
    gradient_x = sobel(imagine, axis=1)
    gradient_y = sobel(imagine, axis=0)

    trasaturi.extend([np.abs(gradient_x).mean(), np.abs(gradient_x).std(), np.abs(gradient_x).max()])
    trasaturi.append([np.abs(gradient_y).mean(), np.abs(gradient_y).std(), np.abs(gradient_y).max()])

    # raport orizontal pe vertical
    trasaturi.append(np.abs(gradient_x).mean() / (np.abs(gradient_x).mean() + 1e-6))

    # histograma orientarilor pe gradient
    # am 8 bin uri pe 0 180 grade
    magnitudine = np.hypot(gradient_x, gradient_y)
    unghi = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180) % 180

    # consider doar gradientii mai puternici
    masca = magnitudine > np.percentile(magnitudine, 75)

    if masca.any():
        histograma, _ = np.histogram(unghi[masca], bins=8, range=(0, 180), weights=magnitudine[masca])
        histograma = histograma / (histograma.sum() + 1e-6)

    else:
        histograma = np.zeros(8)

    trasaturi.extend(histograma.tolist())

    # statisticile globale si percentile
    trasaturi.extend([
        imagine.mean(), imagine.std(), imagine.min(), imagine.max(),
        np.percentile(imagine, 25), np.percentile(imagine, 50), np.percentile(imagine, 75),
        np.percentile(imagine, 90), np.percentile(imagine, 95), np.percentile(imagine, 99),
    ])

    # densitatea de pixeli aprinsi la 5 praguri
    gg = (imagine - imagine.min()) / (imagine.max() - imagine.min() + 1e-6)
    for punct in [70, 80, 85, 90, 95]:
        trasaturi.append((gg > np.percentile(gg, punct)).mean())

    return np.array(trasaturi, dtype=np.float32)

def extrag_manual_set(df, imagine_director, cale_cache):
    # extrag trasaturile manuale pt setul intreg de imagini cu cache

    if os.path.exists(cale_cache):
        print(f" --->> cache: {cale_cache}")
        return np.load(cale_cache)

    X = []
    for i, fid in enumerate(df['id']):
        g = preprocesare(Image.open(os.path.join(imagine_director, fid)))
        X.append(extrag_trasaturi_manuale(g))

        if (i + 1) % 3000 == 0:
            print(f" --> {i + 1} / {len(df)}")

    X = np.vstack(X).astype(np.float32)
    np.save(cale_cache, X)
    return X