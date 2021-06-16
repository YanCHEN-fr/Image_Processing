import cv2
import numpy as np
from matplotlib import pyplot as plt
 
 # calculer difference entre les deux flow optique adjacents
def diffimage(lastframe, nextframe, size):
    diff_frame = nextframe - lastframe
    ABS = abs(diff_frame)
    diff_value = (ABS.sum(axis = 0)).sum(axis = 0)/size
    return diff_frame, diff_value

if __name__ == '__main__':
    cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    ret, lastframe = cap.read()
    lastgray = cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY)
    ret, nextframe = cap.read()
    nextgray = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
    index = 1
    last_diff_value = 0
    lasthist = np.zeros([100, 100])
    a = np.array([0]) # pour preserver deux diffrence entre les deux flow optique adjacents 
    while(ret):
        index += 1
        size = nextframe.size
        flow = cv2.calcOpticalFlowFarneback(lastgray,nextgray,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)  # flow est une flow optique qui a deux channels
        nexthist = cv2.calcHist([flow], [0,1], None, [100,100], [-100,100,-100,100])
        nexthist[nexthist > 255] = 255
        diff_frame, next_diff_value = diffimage(lasthist, nexthist, size)
        a = np.append(a, next_diff_value)
        cv2.imshow('fame', nextframe)

        if (next_diff_value > 0.05 and abs(a[1]-a[0]) < 0.002 ) or next_diff_value > 0.1:
            cv2.imwrite('Frame_%04d.png'%index,nextframe) # conserver image clef
        a = np.delete(a,[0])

        k = cv2.waitKey(15)
        if k == 27:
            break
        lastgray = nextgray
        lasthist = nexthist
        ret, nextframe = cap.read()
        if (ret):
            nextgray = cv2.cvtColor(nextframe, cv2.COLOR_BGR2GRAY)
    cap.realease()
    cv2.destroyAllWindows()