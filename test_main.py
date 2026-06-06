"""
Punct de intrare: ruleaza tot pipeline-ul si genereaza submisiile.

Aceasta este abordarea finala (v7) care a obtinut 0.78545 pe Kaggle.

Pipeline complet:
  1) Incarcare metadate (train.csv, test.csv)
  2) Extragere 52 trasaturi manuale per imagine (cu cache .npy)
  3) Antrenare CNN 5-fold cu 2 seed-uri = 10 modele mediate (out-of-fold)
  4) Predictie test cu TTA + mediere peste folduri
  5) Combinare trasaturi CNN (256) + manuale (52) = 308 trasaturi
  6) SVM RBF cu grid search pe (C, gamma) - Cursul 4
  7) MLP cu grid search pe (arhitectura, alpha) - Cursul 6 / Laboratorul 7
  8) Ensemble: cautare ponderi pe OOF intre CNN + SVM + MLP
  9) Matrici de confuzie pe OOF (pentru raport)
 10) Salvare submisii: ensemble (principala), cnn, svm, mlp

Rulare: python main.py    (recomandat cu GPU - dureaza ~45 min cu GPU)
"""
import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import config
from features import extrage_manual_set
from cnn import LineCNN, infereaza, train_fold


def main():
    # ============================================================
    # 1. INCARCARE METADATE + SETARE SEEDS
    # ============================================================
    train = pd.read_csv(config.TRAIN_CSV)
    test  = pd.read_csv(config.TEST_CSV)
    print(f"train: {len(train)}, test: {len(test)}")

    # setam valorile dinamice in config (folosite de SignalDataset, train_fold)
    config.LABEL_OFFSET = int(train['label'].min())
    config.NR_CLASE     = int(train['label'].nunique())
    print(f"clase: {config.NR_CLASE}, label_offset: {config.LABEL_OFFSET}")

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    if device.type == 'cpu':
        print("ATENTIE: rulezi pe CPU - antrenarea CNN va dura ore intregi.")

    # ============================================================
    # 2. EXTRAGERE TRASATURI MANUALE (cu cache .npy)
    # ============================================================
    print("\n=== Extrag trasaturi manuale ===")
    manual_train = extrage_manual_set(train, config.TRAIN_IMG_DIR,
                                       os.path.join(config.OUTPUT_DIR, "manual_train.npy"))
    manual_test  = extrage_manual_set(test, config.TEST_IMG_DIR,
                                       os.path.join(config.OUTPUT_DIR, "manual_test.npy"))
    print(f"shape: train {manual_train.shape}, test {manual_test.shape}")

    # ============================================================
    # 3. ANTRENARE CNN (2 seed-uri x 5 folduri = 10 modele)
    # ============================================================
    # ponderi de clasa pentru a compensa dezechilibrul (clasa 1 are mai multe exemple)
    w = compute_class_weight('balanced', classes=np.arange(config.NR_CLASE),
                             y=(train['label'] - config.LABEL_OFFSET).values)
    class_weights = torch.tensor(w, dtype=torch.float).to(device)
    print(f"\nponderi clase: {w.round(3)}")

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    oof_probs     = np.zeros((len(train), config.NR_CLASE))
    oof_feats_cnn = np.zeros((len(train), config.FEAT_DIM_CNN))
    oof_labels    = np.zeros(len(train), dtype=int)
    all_paths = []

    for seed in config.SEEDS:
        print(f"\n#### SEED {seed} ####")
        for fold, (tr_idx, val_idx) in enumerate(skf.split(train, train['label'])):
            model_path = os.path.join(config.OUTPUT_DIR, f"m_s{seed}_f{fold}.pth")
            acc = train_fold(model_path, train, tr_idx, val_idx,
                             class_weights, device, seed)
            all_paths.append(model_path)
            print(f"  seed {seed} fold {fold+1}: {acc:.4f}")

            # OOF: extragem predictii pe foldul de validare (model n-a vazut aceste date)
            model = LineCNN(config.NR_CLASE).to(device)
            model.load_state_dict(torch.load(model_path))
            p, f, y = infereaza(model, train.iloc[val_idx], config.TRAIN_IMG_DIR, device)
            oof_probs[val_idx]     += p / len(config.SEEDS)
            oof_feats_cnn[val_idx] += f / len(config.SEEDS)
            oof_labels[val_idx]     = y

    oof_acc_cnn = (oof_probs.argmax(1) == oof_labels).mean()
    print(f"\nOOF CNN acc: {oof_acc_cnn:.4f}")

    # ============================================================
    # 4. PREDICTII TEST CNN (TTA + mediere peste folduri)
    # ============================================================
    print("\n=== Predictii test CNN cu TTA ===")
    test_probs_folds, test_feats_folds, ids = [], [], None
    for path in all_paths:
        model = LineCNN(config.NR_CLASE).to(device)
        model.load_state_dict(torch.load(path))
        p, f, fid = infereaza(model, test, config.TEST_IMG_DIR, device,
                              tta=True, este_test=True)
        test_probs_folds.append(p)
        test_feats_folds.append(f)
        if ids is None:
            ids = fid

    cnn_test_probs = np.mean(test_probs_folds, axis=0)
    cnn_test_feats = np.mean(test_feats_folds, axis=0)

    # ============================================================
    # 5. COMBINARE TRASATURI: CNN (256) + MANUALE (52) = 308
    # ============================================================
    # normalizam separat (StandardScaler - Laboratorul 4), apoi concatenam
    sc_cnn = StandardScaler()
    cnn_oof_norm  = sc_cnn.fit_transform(oof_feats_cnn)
    cnn_test_norm = sc_cnn.transform(cnn_test_feats)

    sc_man = StandardScaler()
    man_oof_norm  = sc_man.fit_transform(manual_train)
    man_test_norm = sc_man.transform(manual_test)

    oof_comb  = np.hstack([cnn_oof_norm,  man_oof_norm])
    test_comb = np.hstack([cnn_test_norm, man_test_norm])
    print(f"\nshape combinat: {oof_comb.shape}, {test_comb.shape}")

    # ============================================================
    # 6. SVM RBF cu grid search (Cursul 4)
    # ============================================================
    print("\n=== SVM (kernel RBF) ===")
    best_svm = (0, 10, 'scale')
    for C in [1, 10, 100]:
        for ga in ['scale', 'auto']:
            s = cross_val_score(SVC(C=C, kernel='rbf', gamma=ga),
                                oof_comb, oof_labels, cv=3, n_jobs=-1).mean()
            print(f"  C={C:<4} gamma={ga:<6}: {s:.4f}")
            if s > best_svm[0]:
                best_svm = (s, C, ga)
    _, bC, bG = best_svm
    print(f"best SVM: C={bC}, gamma={bG}, acc={best_svm[0]:.4f}")

    svm_oof = cross_val_predict(
        SVC(C=bC, kernel='rbf', gamma=bG, probability=True, random_state=config.RANDOM_SEED),
        oof_comb, oof_labels, cv=3, method='predict_proba', n_jobs=-1)
    svm_final = SVC(C=bC, kernel='rbf', gamma=bG, probability=True,
                    random_state=config.RANDOM_SEED)
    svm_final.fit(oof_comb, oof_labels)
    svm_test = svm_final.predict_proba(test_comb)

    # ============================================================
    # 7. MLP cu grid search (Cursul 6 / Laboratorul 7)
    # ============================================================
    print("\n=== MLP (retea neuronala scikit-learn) ===")
    best_mlp = (0, (256,), 1e-3)
    configs_mlp = [
        ((256,), 1e-4), ((256,), 1e-3),
        ((512, 128), 1e-4), ((512, 128), 1e-3),
        ((512, 256, 64), 1e-3),
    ]
    for hl, alpha in configs_mlp:
        s = cross_val_score(
            MLPClassifier(hidden_layer_sizes=hl, alpha=alpha, learning_rate_init=1e-3,
                          max_iter=300, early_stopping=True,
                          random_state=config.RANDOM_SEED),
            oof_comb, oof_labels, cv=3, n_jobs=-1).mean()
        print(f"  hidden={hl} alpha={alpha}: {s:.4f}")
        if s > best_mlp[0]:
            best_mlp = (s, hl, alpha)
    _, bhl, ba = best_mlp
    print(f"best MLP: hidden={bhl}, alpha={ba}, acc={best_mlp[0]:.4f}")

    mlp_oof = cross_val_predict(
        MLPClassifier(hidden_layer_sizes=bhl, alpha=ba, learning_rate_init=1e-3,
                      max_iter=300, early_stopping=True, random_state=config.RANDOM_SEED),
        oof_comb, oof_labels, cv=3, method='predict_proba', n_jobs=-1)
    mlp_final = MLPClassifier(hidden_layer_sizes=bhl, alpha=ba, learning_rate_init=1e-3,
                              max_iter=300, early_stopping=True,
                              random_state=config.RANDOM_SEED)
    mlp_final.fit(oof_comb, oof_labels)
    mlp_test = mlp_final.predict_proba(test_comb)

    # ============================================================
    # 8. ENSEMBLE: cautare ponderi pe OOF
    # ============================================================
    print("\n=== Cautare ponderi ensemble pe OOF ===")
    best = (0, (1, 0, 0))
    grid = np.arange(0, 1.01, 0.05)
    for wc in grid:
        for ws in grid:
            wm = 1 - wc - ws
            if wm < -1e-9:
                continue
            pr = wc * oof_probs + ws * svm_oof + wm * mlp_oof
            acc = (pr.argmax(1) == oof_labels).mean()
            if acc > best[0]:
                best = (acc, (round(wc, 2), round(ws, 2), round(wm, 2)))
    oof_acc_ens, (wc, ws, wm) = best
    print(f"ponderi: CNN={wc} SVM={ws} MLP={wm} -> OOF acc={oof_acc_ens:.4f}")

    ens_test = wc * cnn_test_probs + ws * svm_test + wm * mlp_test
    pred_ens = ens_test.argmax(1)        + config.LABEL_OFFSET
    pred_cnn = cnn_test_probs.argmax(1)  + config.LABEL_OFFSET
    pred_svm = svm_test.argmax(1)        + config.LABEL_OFFSET
    pred_mlp = mlp_test.argmax(1)        + config.LABEL_OFFSET

    # ============================================================
    # 9. MATRICI DE CONFUZIE PE OOF (pentru raport)
    # ============================================================
    print("\n=== MATRICI DE CONFUZIE (OOF) ===")
    ens_oof = (wc * oof_probs + ws * svm_oof + wm * mlp_oof).argmax(1)
    for nume, pr in [("CNN",            oof_probs.argmax(1)),
                     ("SVM CNN+manual", svm_oof.argmax(1)),
                     ("MLP CNN+manual", mlp_oof.argmax(1)),
                     ("ENSEMBLE",       ens_oof)]:
        print(f"\n== {nume} (acc = {accuracy_score(oof_labels, pr):.4f}) ==")
        print(confusion_matrix(oof_labels, pr))
    print("\nRaport ENSEMBLE:")
    print(classification_report(oof_labels, ens_oof,
          target_names=[str(c) for c in sorted(train['label'].unique())]))

    # ============================================================
    # 10. SALVARE SUBMISII
    # ============================================================
    print("\n=== SALVEZ SUBMISIILE ===")
    submisii = [("ensemble", pred_ens), ("cnn", pred_cnn),
                ("svm", pred_svm),       ("mlp", pred_mlp)]
    for nume, pred in submisii:
        path = os.path.join(config.OUTPUT_DIR, f"submission_{nume}.csv")
        pd.DataFrame({"id": ids, "label": pred}).to_csv(path, index=False)
        distrib = pd.Series(pred).value_counts().sort_index().to_dict()
        print(f"  {path} | distributie: {distrib}")

    print("\nGATA. Submisia principala (cea care a obtinut 0.78545 pe Kaggle):")
    print(f"  {os.path.join(config.OUTPUT_DIR, 'submission_ensemble.csv')}")


if __name__ == "__main__":
    main()