import cv2
import  numpy as np
import operator
from functools import reduce

def pre_process(lastframe):
    a_last = np.zeros(127, np.int16)
    lasthsv = cv2.cvtColor(lastframe,cv2.COLOR_BGR2HSV)
    h = (lasthsv[:,:,0]*359)/lasthsv[:,:,0].max() # mapping h(0, 179) -> h(0, 360)
    s = lasthsv[:,:,1]/lasthsv[:,:,1].max() # mapping s(0, 255) -> s(0, 1) 

    H = ((h%350 + 10)//20).astype(np.float32) # mapping h -> H
    S = (abs(s-0.000000000000001)//0.125).astype(np.float32) # mapping s -> S

    N = S*18 + H - 17 # mapping H, S -> HS
    N[N<0] = 0
    a_last = (cv2.calcHist([N], [0], None, [127],[0, 126])).astype(np.int16) # compter nombre de pixel dans HS par chaque frame
    a_last = a_last.flatten()
    return a_last

# Définir les seuils
V1 = 0.35
V2 = 0.6

image_clef_index = 1 # noter image-clef index dérnier
index = 1 # index pour frame lu
cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
ret, lastframe = cap.read()
cv2.imwrite('Frame_%04d.png'%index,lastframe)
a_last = np.zeros(127, int) # définir un array pour HS
a_list = np.zeros([1, 127]) # définir un matrix pour une série de HS de toutes les frames lu
Similar2 = np.array([3]) # définir un array pour une série de S(F_(framep pris), F_n)

while (ret):
    index += 1
    ret, nextframe = cap.read()
    a_next = pre_process(nextframe)
    a_list = np.append(a_list, [a_next], axis = 0)
    Similar = 1/(2*nextframe.shape[0]*nextframe.shape[1]) * sum(abs(a_next - a_last)) # calculer la similarité 
    if Similar < V1:
        pass
    elif Similar > V2:
        print("image clef")
        cv2.imwrite('Frame_%04d.png'%index,nextframe)
        image_clef_index = index

    # détection de segmentation
    elif V1 < Similar and Similar < V2:
        if ((index - image_clef_index) > 16):
            i = image_clef_index + 15
            aa = 0
            while (i < index):
                Similar2 = np.append(Similar2, [1/(2*nextframe.shape[0]*nextframe.shape[1]) * sum(abs(a_next - a_list[i]))], axis = 0)
                i += 15
            aa = np.where(Similar2 < V1)
            if aa != 0:
                print("image clef2")
                cv2.imwrite('Frame_%04d.png'%index,nextframe)
                image_clef_index = index
            
    lastframe = nextframe
    a_last = a_next

    cv2.imshow('Frame', nextframe)
    k = cv2.waitKey(15)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()



