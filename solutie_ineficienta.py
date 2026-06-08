import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import configurare
from features import extrag_manual_set


def main():
    # incarcare metadate
    date_antrenare = pd.read_csv(configurare.TRAIN_CSV_FISIER)
    date_test      = pd.read_csv(configurare.TEST_CSV_FISIER)

    print(f"train: {len(date_antrenare)}, test: {len(date_test)}")

    # mapez etichetele 1 5 la 0 4 pentru clasificatori

    configurare.LABEL_OFFSET = int(date_antrenare['label'].min())
    configurare.NR_CLASE     = int(date_antrenare['label'].nunique())
    etichete_antrenare = (date_antrenare['label'] - configurare.LABEL_OFFSET).values
    np.random.seed(configurare.RANDOM_SEED)

    # extrag trasaturile manuale
    print()
    print("EXTRAG TRASATURILE MANUALE")
    trasaturi_train = extrag_manual_set(date_antrenare, configurare.TRAIN_IMAGINI_DIRECTOR,
                                        os.path.join(configurare.OUTPUT_DIRECTOR, "manual_train.npy"))
    trasaturi_test  = extrag_manual_set(date_test, configurare.TEST_IMAGINI_DIRECTOR,
                                        os.path.join(configurare.OUTPUT_DIRECTOR, "manual_test.npy"))

    print(f"shape: train {trasaturi_train.shape}, test {trasaturi_test.shape}")

    # normalizare
    scalator = StandardScaler()
    trasaturi_train_norm = scalator.fit_transform(trasaturi_train)
    trasaturi_test_norm  = scalator.transform(trasaturi_test)

    # split stratificat pentru validare
    trasaturi_tr, trasaturi_val, etichete_tr, etichete_val = train_test_split(
        trasaturi_train_norm, etichete_antrenare, test_size=0.15, stratify=etichete_antrenare,
        random_state=configurare.RANDOM_SEED)

    print(f"train intern: {len(etichete_tr)}, validare: {len(etichete_val)}")
    id_uri_test = date_test['id'].tolist()

    # model 1: naive bayes
    print()
    print("NAIVE BAYES")
    nb = GaussianNB()
    nb.fit(trasaturi_tr, etichete_tr)
    acuratete_nb = accuracy_score(etichete_val, nb.predict(trasaturi_val))
    print(f"acuratete validare: {acuratete_nb:.4f}")

    # model 2: knn -->> cautare k optim prin cross-validation
    print()
    print("KNN -->> CAUTARE K")
    cel_mai_bun_k = (0, 5)

    for k in [3, 5, 7, 11, 15, 21, 31]:
        scor = cross_val_score(
            KNeighborsClassifier(n_neighbors=k, weights='distance'),
            trasaturi_tr, etichete_tr, cv=3, n_jobs=-1).mean()

        print(f"  K={k:2d}: CV acc = {scor:.4f}")

        if scor > cel_mai_bun_k[0]:
            cel_mai_bun_k = (scor, k)

    _, k_optim = cel_mai_bun_k
    knn = KNeighborsClassifier(n_neighbors=k_optim, weights='distance')
    knn.fit(trasaturi_tr, etichete_tr)
    acuratete_knn = accuracy_score(etichete_val, knn.predict(trasaturi_val))
    print(f"best K = {k_optim}, acuratete validare: {acuratete_knn:.4f}")

    # model 3: svm cu kernel rbf
    print()
    print("SVM -->> GIRD SEARCH C, GAMMA")

    cel_mai_bun_svm = (0, 1, 'scale')

    for C in [1, 10, 100]:
        for gamma in ['scale', 'auto']:
            scor = cross_val_score(
                SVC(C=C, kernel='rbf', gamma=gamma),
                trasaturi_tr, etichete_tr, cv=3, n_jobs=-1).mean()

            print(f"  C={C:<4} gamma={gamma:<6}: CV acc = {scor:.4f}")

            if scor > cel_mai_bun_svm[0]:
                cel_mai_bun_svm = (scor, C, gamma)

    _, cel_mai_bun_C, cel_mai_bun_gamma = cel_mai_bun_svm

    svm = SVC(C=cel_mai_bun_C, kernel='rbf', gamma=cel_mai_bun_gamma, probability=True,
              random_state=configurare.RANDOM_SEED)
    svm.fit(trasaturi_tr, etichete_tr)
    acuratete_svm = accuracy_score(etichete_val, svm.predict(trasaturi_val))
    print(f"best SVM: C={cel_mai_bun_C}, gamma={cel_mai_bun_gamma}, acuratete validare: {acuratete_svm:.4f}")

    # model 4: mlp grid pe arhitectura si regularizare alpha
    print()
    print("MLP")

    cel_mai_bun_mlp = (0, (256,), 1e-3)
    configuratii_mlp = [
        ((128,),         1e-4), ((128,),         1e-3),
        ((256,),         1e-4), ((256,),         1e-3),
        ((512, 128),     1e-4), ((512, 128),     1e-3),
        ((512, 256, 64), 1e-3),
    ]

    for straturi_ascunse, alpha in configuratii_mlp:
        scor = cross_val_score(
            MLPClassifier(hidden_layer_sizes=straturi_ascunse, alpha=alpha,
                          learning_rate_init=1e-3, max_iter=300,
                          early_stopping=True, random_state=configurare.RANDOM_SEED),
            trasaturi_tr, etichete_tr, cv=3, n_jobs=-1).mean()

        print(f"  hidden={str(straturi_ascunse):<18} alpha={alpha}: CV acc = {scor:.4f}")

        if scor > cel_mai_bun_mlp[0]:
            cel_mai_bun_mlp = (scor, straturi_ascunse, alpha)

    _, cel_mai_bun_hl, cel_mai_bun_alpha = cel_mai_bun_mlp

    mlp = MLPClassifier(hidden_layer_sizes=cel_mai_bun_hl, alpha=cel_mai_bun_alpha,
                        learning_rate_init=1e-3, max_iter=300,
                        early_stopping=True, random_state=configurare.RANDOM_SEED)

    mlp.fit(trasaturi_tr, etichete_tr)
    acuratete_mlp = accuracy_score(etichete_val, mlp.predict(trasaturi_val))
    print(f"best MLP: hidden={cel_mai_bun_hl}, alpha={cel_mai_bun_alpha}, acuratete validare: {acuratete_mlp:.4f}")

    # ensemble prin soft voting --->>> cautare ponderi pe validare
    prob_nb  = nb.predict_proba(trasaturi_val)
    prob_knn = knn.predict_proba(trasaturi_val)
    prob_svm = svm.predict_proba(trasaturi_val)
    prob_mlp = mlp.predict_proba(trasaturi_val)
    folosim_nb = acuratete_nb >= 0.40

    print()
    print(f"NB inclus in ensemble: {folosim_nb}")

    print()
    print("CAUTARE PONDERI ENSEMBLE PE VALIDARE")

    cel_mai_bun_ensemble = (0, None)

    grila_ponderi = np.arange(0, 1.01, 0.1)

    for pondere_nb in grila_ponderi:
        if not folosim_nb and pondere_nb > 0:
            continue

        for pondere_knn in grila_ponderi:
            for pondere_svm in grila_ponderi:
                pondere_mlp = 1 - pondere_nb - pondere_knn - pondere_svm

                if pondere_mlp < -1e-9 or pondere_mlp > 1.001:
                    continue

                prob_ensemble = pondere_nb*prob_nb + pondere_knn*prob_knn + pondere_svm*prob_svm + pondere_mlp*prob_mlp
                acuratete = accuracy_score(etichete_val, prob_ensemble.argmax(1))

                if acuratete > cel_mai_bun_ensemble[0]:
                    cel_mai_bun_ensemble = (acuratete, (round(pondere_nb, 2), round(pondere_knn, 2), round(pondere_svm, 2), round(pondere_mlp, 2)))

    acuratete_ensemble, (pondere_nb, pondere_knn, pondere_svm, pondere_mlp) = cel_mai_bun_ensemble

    print(f"ponderi: NB={pondere_nb} KNN={pondere_knn} SVM={pondere_svm} MLP={pondere_mlp}")

    print(f"acuratete ensemble pe validare: {acuratete_ensemble:.4f}")

    # matrici de confuzie pe validare
    predictii_ensemble_val = (pondere_nb*prob_nb + pondere_knn*prob_knn + pondere_svm*prob_svm + pondere_mlp*prob_mlp).argmax(1)

    print()
    print("MATRICI DE CONFUZIE")
    for nume, predictii_model in [("Naive Bayes", nb.predict(trasaturi_val)),
                                   ("KNN",         knn.predict(trasaturi_val)),
                                   ("SVM",         svm.predict(trasaturi_val)),
                                   ("MLP",         mlp.predict(trasaturi_val)),
                                   ("ENSEMBLE",    predictii_ensemble_val)]:
        print(f"\n== {nume} (acc = {accuracy_score(etichete_val, predictii_model):.4f}) ==")
        print(confusion_matrix(etichete_val, predictii_model))

    print()
    print("Raport ENSEMBLE:")
    print(classification_report(etichete_val, predictii_ensemble_val,
            target_names=[str(c) for c in sorted(date_antrenare['label'].unique())]))

    # reantrenare pe tot setul de train + predictii test
    # refacem fiecare model pe TOT setul de train ca sa folosim toate exemplele inainte de a prezice testul
    print()
    print("REANTRENARE PE TOT SETUL SI PREDICTII TEST")

    nb_final  = GaussianNB().fit(trasaturi_train_norm, etichete_antrenare)
    knn_final = KNeighborsClassifier(n_neighbors=k_optim, weights='distance').fit(trasaturi_train_norm, etichete_antrenare)
    svm_final = SVC(C=cel_mai_bun_C, kernel='rbf', gamma=cel_mai_bun_gamma, probability=True,
                    random_state=configurare.RANDOM_SEED).fit(trasaturi_train_norm, etichete_antrenare)

    mlp_final = MLPClassifier(hidden_layer_sizes=cel_mai_bun_hl, alpha=cel_mai_bun_alpha,
                              learning_rate_init=1e-3, max_iter=300,
                              early_stopping=True,
                              random_state=configurare.RANDOM_SEED).fit(trasaturi_train_norm, etichete_antrenare)

    prob_test_nb  = nb_final.predict_proba(trasaturi_test_norm)
    prob_test_knn = knn_final.predict_proba(trasaturi_test_norm)
    prob_test_svm = svm_final.predict_proba(trasaturi_test_norm)
    prob_test_mlp = mlp_final.predict_proba(trasaturi_test_norm)

    prob_test_ensemble = pondere_nb*prob_test_nb + pondere_knn*prob_test_knn + pondere_svm*prob_test_svm + pondere_mlp*prob_test_mlp

    # convertesc inapoi din 0 4 in 1 5
    submisii = [
        ("ensemble", prob_test_ensemble.argmax(1) + configurare.LABEL_OFFSET),
        ("nb",       prob_test_nb.argmax(1)        + configurare.LABEL_OFFSET),
        ("knn",      prob_test_knn.argmax(1)        + configurare.LABEL_OFFSET),
        ("svm",      prob_test_svm.argmax(1)        + configurare.LABEL_OFFSET),
        ("mlp",      prob_test_mlp.argmax(1)        + configurare.LABEL_OFFSET),
    ]

    print()
    print("SALVEZ SUBMISIILE CLASICE")
    for nume, pred in submisii:
        cale_fisier = os.path.join(configurare.OUTPUT_DIRECTOR, f"submission_clasic_{nume}.csv")
        pd.DataFrame({"id": id_uri_test, "label": pred}).to_csv(cale_fisier, index=False)
        distributie = pd.Series(pred).value_counts().sort_index().to_dict()
        print(f"  {cale_fisier} | distributie: {distributie}")

    print("REZUMAT SCORURI VALIDARE")
    print(f"  Naive Bayes  : {acuratete_nb:.4f}")
    print(f"  KNN (K={k_optim})   : {acuratete_knn:.4f}")
    print(f"  SVM          : {acuratete_svm:.4f}")
    print(f"  MLP          : {acuratete_mlp:.4f}")
    print(f"  ENSEMBLE     : {acuratete_ensemble:.4f}")


if __name__ == "__main__":
    main()
