# preprocesez imaginile ca sa pot scoate obiectele reprezentate de linii din zgomotul de pe fundal
# imaginile mele sunt de fapt spectograme RGBA pe un fundal mov
# liniile sunt linii subtiri cu diferita orientare

# o sa fac convertire la luminanta cu canalul L
# scot liniile slabe cu contrast stretching la percentilele de la 1 la 99
# normalizez per imagine ca sa uniformizez intensitatea

import numpy as np

def preprocesare(imagine):
    # o sa fac convertire la luminanta cu canalul L
    matrice = np.asarray(imagine.convert('L'), dtype=np.float32)

    # scot liniile slabe cu contrast stretching la percentilele de la 1 la 99
    percentil_minim, percentil_maxim = np.percentile(matrice, 1), np.percentile(matrice, 99)

    if percentil_maxim > percentil_minim:
        matrice = np.clip((matrice - percentil_minim)/(percentil_maxim - percentil_minim), 0, 1)
    else:
        matrice = matrice - matrice.min()

    # normalizez per imagine ca sa uniformizez intensitatea
    matrice = (matrice - matrice.mean()) / (matrice.std() + 1e-6)

    # returnez array float32 cu imaginea preprocesata
    return matrice
