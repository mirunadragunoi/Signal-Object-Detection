# preprocesez imaginile ca sa pot scoate obiectele reprezentate de linii din zgomotul de pe fundal
# imaginile mele sunt de fapt spectograme RGBA pe un fundal mov
# liniile sunt linii subtiri cu diferita orientare

# o sa fac convertire la luminanta cu canalul L
# scot liniile slabe cu contrast stretching la percentilele de la 1 la 99
# normalizez per imagine ca sa uniformizez intensitatea

import numpy as np

def preprocesare(imagine):
    # o sa fac convertire la luminanta cu canalul L
    img = np.asarray(imagine.convert('L'), dtype=np.float32)

    # scot liniile slabe cu contrast stretching la percentilele de la 1 la 99
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)

    if hi > lo:
        img = np.clip((img - lo)/(hi - lo), 0, 1)
    else:
        img = img - img.min()

    # normalizez per imagine ca sa uniformizez intensitatea
    img = (img - img.mean()) / (img.std() + 1e-6)

    # returnez array float32 cu imaginea preprocesata
    return img