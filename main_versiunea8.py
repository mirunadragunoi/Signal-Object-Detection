import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import configurare
from features import extrag_manual_set
from cnn import LineCNN, infereaza, train_pe_fold

def main():
    # fortez ca sa folosesc 4 seed uri pt versiunea 8
    configurare.SEEDS = [42, 123, 2024, 7]

    # incarc metadatele si setez seed urile
    date_antrenare = pd.read_csv(configurare.TRAIN_CSV_FISIER)
    date_test = pd.read_csv(configurare.TEST_CSV_FISIER)

    print(f"train/antrenare: {len(date_antrenare)}, test: {len(date_test)}")

    # setez valorile dinamice in configurare
    configurare.LABEL_OFFSET = int(date_antrenare['label'].min())
    configurare.NR_CLASE = int(date_antrenare['label'].nunique())

    print(f"clase: {configurare.NR_CLASE}, label offset: {configurare.LABEL_OFFSET}")


    np.random.seed(configurare.RANDOM_SEED)
    torch.manual_seed(configurare.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    if device.type == "cpu":
        print("ATENTIE RULARE PE CPU!")

    # extrag trasaturile manuale
    print()
    print("EXTRAGERE TRASATURI MANUALE")

    trasaturi_manuale_train = extrag_manual_set(date_antrenare, configurare.TRAIN_IMAGINI_DIRECTOR,
                              os.path.join(configurare.OUTPUT_DIRECTOR, "trasaturi_manuale_train.npy"))

    trasaturi_manuale_test = extrag_manual_set(date_test, configurare.TEST_IMAGINI_DIRECTOR,
                              os.path.join(configurare.OUTPUT_DIRECTOR, "trasaturi_manuale_test.npy"))

    print(f"SHAPE: train {trasaturi_manuale_train}, test {trasaturi_manuale_test}")

    # antrenez efectiv CNN ul
    # 4 seed uri cu 5 fold uri deci 20 de modele

    # ponderi de clasa pt compensarea dezechilibrului
    ponderi = compute_class_weight('balanced', classes=np.arange(configurare.NR_CLASE), y=(date_antrenare['label'] - configurare.LABEL_OFFSET).values)

    ponderi_clase = torch.tensor(ponderi, dtype=torch.float32).to(device)

    print()
    print(f"ponderi clase: {ponderi.round(3)}")

    impartire_stratificata = StratifiedKFold(n_splits=configurare.NR_FOLDS, shuffle=True, random_state=42)
    probabilitati_oof = np.zeros((len(date_antrenare), configurare.NR_CLASE))
    trasaturi_oof_cnn = np.zeros((len(date_antrenare), configurare.DIMENSIUNE_CNN))
    etichete_oof = np.zeros(len(date_antrenare), dtype=int)
    cai_modele = []

    for seed in configurare.SEEDS:
        print()
        print(f" SEED {seed}")
        for fold, (train_index, valoare_index) in enumerate(impartire_stratificata.split(date_antrenare, date_antrenare['label'])):
            model_cale = os.path.join(configurare.OUTPUT_DIRECTOR, f"m_s{seed}_f{fold}.pth")
            acuratete = train_pe_fold(model_cale, date_antrenare, train_index, valoare_index,
                             ponderi_clase, device, seed)

            cai_modele.append(model_cale)

            print(f"  seed {seed} fold {fold+1}: {acuratete:.4f}")

            # OOF -->> extragem predictii pe foldul de validare
            model = LineCNN(configurare.NR_CLASE).to(device)
            model.load_state_dict(torch.load(model_cale))
            probabilitati_val, trasaturi_val, etichete_val = infereaza(model, date_antrenare.iloc[valoare_index], configurare.TRAIN_IMAGINI_DIRECTOR, device)

            probabilitati_oof[valoare_index] += probabilitati_val / len(configurare.SEEDS)
            trasaturi_oof_cnn[valoare_index] += trasaturi_val / len(configurare.SEEDS)
            etichete_oof[valoare_index] = etichete_val

    acuratete_oof_cnn = (probabilitati_oof.argmax(1) == etichete_oof).mean()
    print()
    print(f"OOF CNN acc (versiunea 8 de cod): {acuratete_oof_cnn:.4f}")


    # predictii test CNN cu TTA si mediere peste folduri
    print()
    print("PREDICTII TEST CNN CU TTA (versiunea 8 cu 20 de modele)")

    test_probabilitati_folds, test_trasaturi_folds, id_uri = [], [], None

    for cale in cai_modele:
        model = LineCNN(configurare.NR_CLASE).to(device)
        model.load_state_dict(torch.load(cale))

        probabilitati_fold, trasaturi_fold, id_uri_fold = infereaza(model, date_test, configurare.TEST_IMAGINI_DIRECTOR, device, tta=True, este_test=True)

        test_probabilitati_folds.append(probabilitati_fold)
        test_trasaturi_folds.append(trasaturi_fold)

        if id_uri is None:
            id_uri = id_uri_fold

    probabilitati_test_cnn = np.mean(test_probabilitati_folds, axis=0)
    trasaturi_test_cnn = np.mean(test_trasaturi_folds, axis=0)


    # combinare trasaturi de la cnn plus alea manuale

    # normalizez prima oara separat apoi concanetez
    scalator_cnn = StandardScaler()
    cnn_oof_normalizat = scalator_cnn.fit_transform(trasaturi_oof_cnn)
    cnn_test_normalizat = scalator_cnn.transform(trasaturi_test_cnn)

    scalator_manual = StandardScaler()
    manual_oof_normalizat = scalator_manual.fit_transform(trasaturi_manuale_train)
    manual_test_normalizat = scalator_manual.transform(trasaturi_manuale_test)

    oof_combinat = np.hstack([cnn_oof_normalizat, manual_oof_normalizat])
    test_combinat = np.hstack([cnn_test_normalizat, manual_test_normalizat])
    print()
    print(f"shape combinat: {oof_combinat.shape}, {test_combinat.shape}")

    # SVM RBF cu grid search
    print()
    print("SVM CU KERNEL RBF)")
    cel_mai_bun_svm = (0, 10, 'scale')
    for C in [1, 10, 100]:
        for gamma in ['scale', 'auto']:
            scor = cross_val_score(SVC(C=C, kernel='rbf', gamma=gamma), oof_combinat, etichete_oof, cv=3, n_jobs=-1).mean()
            print(f"  C={C:<4} gamma={gamma:<6}: {scor:.4f}")
            if scor > cel_mai_bun_svm[0]:
                cel_mai_bun_svm = (scor, C, gamma)
    _, cel_mai_bun_C, cel_mai_bun_gamma = cel_mai_bun_svm
    print(f"best SVM: C={cel_mai_bun_C}, gamma={cel_mai_bun_gamma}, acc={cel_mai_bun_svm[0]:.4f}")

    svm_oof = cross_val_predict(
        SVC(C=cel_mai_bun_C, kernel='rbf', gamma=cel_mai_bun_gamma, probability=True, random_state=configurare.RANDOM_SEED),
        oof_combinat, etichete_oof, cv=3, method='predict_proba', n_jobs=-1)

    svm_final = SVC(C=cel_mai_bun_C, kernel='rbf', gamma=cel_mai_bun_gamma, probability=True, random_state=configurare.RANDOM_SEED)

    svm_final.fit(oof_combinat, etichete_oof)
    svm_test = svm_final.predict_proba(test_combinat)

    # MLP CU GRID SEARCH
    print()
    print("RETEA NEURONALA")

    cel_mai_bun_mlp = (0, (256,), 1e-3)
    configuratii_mlp = [
        ((256,), 1e-4), ((256,), 1e-3),
        ((512, 128), 1e-4), ((512, 128), 1e-3),
        ((512, 256, 64), 1e-3),
    ]

    for straturi_ascunse, alpha in configuratii_mlp:
        scor = cross_val_score(
            MLPClassifier(hidden_layer_sizes=straturi_ascunse, alpha=alpha, learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                          random_state=configurare.RANDOM_SEED), oof_combinat, etichete_oof, cv=3, n_jobs=-1).mean()

        print(f"  hidden={straturi_ascunse} alpha={alpha}: {scor:.4f}")

        if scor > cel_mai_bun_mlp[0]:
            cel_mai_bun_mlp = (scor, straturi_ascunse, alpha)

    _, cel_mai_bun_hl, cel_mai_bun_alpha = cel_mai_bun_mlp
    print(f"best MLP: hidden={cel_mai_bun_hl}, alpha={cel_mai_bun_alpha}, acc={cel_mai_bun_mlp[0]:.4f}")

    mlp_oof = cross_val_predict(
        MLPClassifier(hidden_layer_sizes=cel_mai_bun_hl, alpha=cel_mai_bun_alpha, learning_rate_init=1e-3, max_iter=300, early_stopping=True, random_state=configurare.RANDOM_SEED),
        oof_combinat, etichete_oof, cv=3, method='predict_proba', n_jobs=-1)

    mlp_final = MLPClassifier(hidden_layer_sizes=cel_mai_bun_hl, alpha=cel_mai_bun_alpha, learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                              random_state=configurare.RANDOM_SEED)

    mlp_final.fit(oof_combinat, etichete_oof)
    mlp_test = mlp_final.predict_proba(test_combinat)

    # special pt versiunea de cod 8
    # adaug un knn ca al patrulea model in ensemble
    print()
    print("KNN - CAUTARE K OPTIM")

    cel_mai_bun_knn = (0, 11)
    for k in [3, 5, 7, 11, 15, 21, 31]:
        scor = cross_val_score(KNeighborsClassifier(n_neighbors=k, weights='distance'), oof_combinat, etichete_oof, cv=3, n_jobs=-1).mean()

        print(f"  K={k:2d}: CV acc = {scor:.4f}")

        if scor > cel_mai_bun_knn[0]:
            cel_mai_bun_knn = (scor, k)

    _, k_optim = cel_mai_bun_knn
    print(f"best KNN: K={k_optim}, acc={cel_mai_bun_knn[0]:.4f}")

    knn_oof = cross_val_predict(
        KNeighborsClassifier(n_neighbors=k_optim, weights='distance'),
        oof_combinat, etichete_oof, cv=3, method='predict_proba', n_jobs=-1)

    knn_final = KNeighborsClassifier(n_neighbors=k_optim, weights='distance')
    knn_final.fit(oof_combinat, etichete_oof)
    knn_test = knn_final.predict_proba(test_combinat)

    # ensemble cu toate 4 modele caut ponderi pe oof
    print()
    print("CAUTARE PONDERI ENSEMBLE PE OOF PE 4 MODELE")

    cel_mai_bun_ensemble = (0, (1, 0, 0, 0))
    grila_ponderi = np.arange(0, 1.01, 0.05)

    for pondere_cnn in grila_ponderi:
        for pondere_svm in grila_ponderi:
            for pondere_mlp in grila_ponderi:
                pondere_knn = 1 - pondere_cnn - pondere_svm - pondere_mlp
                if pondere_knn < -1e-9 or pondere_knn > 1.001:
                    continue

                prob_ensemble = pondere_cnn * probabilitati_oof + pondere_svm * svm_oof + pondere_mlp * mlp_oof + pondere_knn * knn_oof

                acuratete = (prob_ensemble.argmax(1) == etichete_oof).mean()

                if acuratete > cel_mai_bun_ensemble[0]:
                    cel_mai_bun_ensemble = (acuratete, (round(pondere_cnn, 2), round(pondere_svm, 2), round(pondere_mlp, 2), round(pondere_knn, 2)))

    acuratete_oof_ensemble, (pondere_cnn, pondere_svm, pondere_mlp, pondere_knn) = cel_mai_bun_ensemble
    print(f"ponderi: CNN = {pondere_cnn} SVM = {pondere_svm} MLP = {pondere_mlp} KNN = {pondere_knn} -->>> OOF acc = {acuratete_oof_ensemble:.4f}")

    test_ensemble = pondere_cnn * probabilitati_test_cnn + pondere_svm * svm_test + pondere_mlp * mlp_test + pondere_knn * knn_test
    predictii_ensemble = test_ensemble.argmax(1) + configurare.LABEL_OFFSET
    predictii_cnn = probabilitati_test_cnn.argmax(1) + configurare.LABEL_OFFSET
    predictii_svm = svm_test.argmax(1) + configurare.LABEL_OFFSET
    predictii_mlp = mlp_test.argmax(1) + configurare.LABEL_OFFSET
    predictii_knn = knn_test.argmax(1) + configurare.LABEL_OFFSET


    # matrici de confuzie pe oof
    print()
    print("MATRICI DE CONFUZIE (OOF) PT VERSIUNEA 8")

    oof_ensemble = (pondere_cnn * probabilitati_oof + pondere_svm * svm_oof + pondere_mlp * mlp_oof + pondere_knn * knn_oof).argmax(1)

    for nume, predictii_model in [("CNN", probabilitati_oof.argmax(1)),
                     ("SVM CNN+manual", svm_oof.argmax(1)),
                     ("MLP CNN+manual", mlp_oof.argmax(1)),
                     ("KNN CNN+manual", knn_oof.argmax(1)),
                     ("ENSEMBLE", oof_ensemble)]:
        print(f"\n {nume} (acc = {accuracy_score(etichete_oof, predictii_model):.4f}) ==")
        print(confusion_matrix(etichete_oof, predictii_model))

    print()
    print("Raport ENSEMBLE:")
    print(classification_report(etichete_oof, oof_ensemble, target_names=[str(c) for c in sorted(date_antrenare['label'].unique())]))

    # salvare submisii
    print()
    print("SALVARE SUBMISII VERSIUNE 8")
    submisii = [("ensemble", predictii_ensemble), ("cnn", predictii_cnn), ("svm", predictii_svm), ("mlp", predictii_mlp), ("knn", predictii_knn)]

    for nume, pred in submisii:
        cale_fisier = os.path.join(configurare.OUTPUT_DIRECTOR, f"submission_v8_{nume}.csv")
        pd.DataFrame({"id": id_uri, "label": pred}).to_csv(cale_fisier, index=False)
        distributie = pd.Series(pred).value_counts().sort_index().to_dict()
        print(f"  {cale_fisier} | distributie: {distributie}")

    print()
    print("GATA! SUBMISIA PRINCIPALA")
    print(f"  {os.path.join(configurare.OUTPUT_DIRECTOR, 'submission_ensemble_v8.csv')}")



if __name__ == "__main__":
    main()
