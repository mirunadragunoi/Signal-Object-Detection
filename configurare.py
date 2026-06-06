# configurez toate caile si parametrii

import os

# caile pt date
DIRECTOR_DATE = "./data"
TRAIN_CSV_FISIER = os.path.join(DIRECTOR_DATE, "train.csv")
TEST_CSV_FISIER = os.path.join(DIRECTOR_DATE, "test.csv")
TRAIN_IMAGINI_DIRECTOR =  os.path.join(DIRECTOR_DATE, "train")
TEST_IMAGINI_DIRECTOR = os.path.join(DIRECTOR_DATE, "test")

# folderul pt output uri ca sa mi salvez ce modele si features mai am
OUTPUT_DIRECTOR = "./output"
os.makedirs(OUTPUT_DIRECTOR, exist_ok=True)

# imi setez aici hiperparametrii pt modelul cnn pe care l folosesc

# dimensiunea imaginilor
IMAGINE_INALTIME = 128
IMAGINE_LUNGIME = 55

# dimensiune batch
BATCH_DIMENSIUNE = 64

# nr epoci pt antrenarea pe fold
EPOCI = 45

# pt optimizarea de early stopping
RABDARE = 13

# setez nr minim de epoci ca sa nu opresc prea devreme
MIN_EPOCI = 18

# nr de folduri
NR_FOLDS = 5

# cate seed uri pe cate folduri
SEEDS = [42, 123]

# rata de invatare de baza
LR = 1e-3

# maximul de rata de invatare
MAX_LR = 3e-3

# regularizarea cu masura l2 euclidiana
REGULARIZARE_COEF = 1e-4

# folosesc un label smooting in cross entropy loss
NETEZIRE_ETICHETE = 0.05

# dimensiunea trasaturilor extrase din cnn
DIMENSIUNE_CNN = 256



# SEED UL GLOBAL PT REPRODUCTIBILITATE
RANDOM_SEED = 42



# valori pe care le mai setez dinamic in main dupa ce incarc datele
LABEL_OFFSET = None
NR_CLASE = None