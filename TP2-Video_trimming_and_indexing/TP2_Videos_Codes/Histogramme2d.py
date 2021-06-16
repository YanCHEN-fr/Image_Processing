# 画2d颜色彩色直方图
'''
import cv2
import numpy as np  
from time import clock
import sys
#import cv2.video

if __name__ == '__main__':
    yuv_map = np.zeros((240,240,3),np.uint8)
    u,v = np.indices(yuv_map.shape[:2])
    yuv_map[:,:,0] = u
    yuv_map[:,:,1] = v
    yuv_map[:,:,2] = 235
    yuv_map = cv2.cvtColor(yuv_map, cv2.COLOR_YUV2BGR)
    cv2.imshow('hsv_map', yuv_map)

    cv2.namedWindow('hist',0)
    hist_scale = 10
    def set_scale(val):
        global hist_scale
        hist_scale = val
    cv2.createTrackbar('scale', 'hist', hist_scale, 32, set_scale)

    try:fn = sys.argv[1]
    except:fn = 0
    cam = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    while(True):
        flag,frame = cam.read()
        cv2.imshow('camera', frame)
        small = cv2.pyrDown(frame)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        dark = yuv[:,:,2]<32
        yuv[dark] = 0
        #h = cv2.calcHist([yuv], [0,1], None, [240,240], [16,240,16,240])
        h = cv2.calcHist([yuv], [1], None, [240], [16,240])
        h = np.clip(h*0.005*hist_scale, 0, 1)
        vis = yuv_map*h[:,:,np.newaxis] / 255.0
        cv2.imshow('hist', vis)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
'''

#画2d颜色直方图（黑白）
'''
import cv2
import numpy as np

#Ouverture du flux video
cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

while(True):
    ret, frame1 = cap.read() # Passe à l'image suivante
    yuv = cv2.cvtColor(frame1,cv2.COLOR_BGR2YUV)
    #hist = cv2.calcHist([hsv2], [0,1], None, [240,240], [16,240,16,240])
    hist = cv2.calcHist([yuv], [0,1], None, [240,240], [16,240,16,240])
    cv2.imshow('histogramme 2d', hist)
    k = cv2.waitKey(15)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''

# affichier flow vx, vy
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Ouverture du flux video
cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

while(ret):
    index += 1
    yuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV)
    flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)  # flow est une flow optique qui a deux channels
    hist = cv2.calcHist([flow], [0,1], None, [100,100], [-100,100,-100,100])
    cv2.imshow('Image et Champ de vitesses (Farneback)',frame2)
    cv2.imshow('histogramme 2d', hist)

    k = cv2.waitKey(15) & 0xff

    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        #cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
        cv2.imwrite('histogramme_%04d.png'%index,hist)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
cap.release()
cv2.destroyAllWindows()



# 计算稠密光流 + 画视频2d颜色直方图（黑白）
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Ouverture du flux video
cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

while(ret):
    index += 1
    yuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV)
    hist = cv2.calcHist([yuv], [1,2], None, [240,240], [0,240, 0, 240])
    flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)  # flow est une flow optique qui a deux channels
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme    amax(mag): le maximum dans le matrix mag

    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2,bgr)) # combiner frame2 et bgr à l'orientation verticale

    cv2.imshow('Image et Champ de vitesses (Farneback)',result.astype(np.uint8))
    cv2.imshow('histogramme 2d', hist)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
        cv2.imwrite('histogramme_%04d.png'%index,hist)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
print(a.max())
print(b.max())
cap.release()
cv2.destroyAllWindows()
'''

'''
# 计算稠密光流 + 画视频-维颜色直方图（黑白）
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Ouverture du flux video
cap = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")

ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
a = []
b = []

while(ret):
    index += 1
    yuv = cv2.cvtColor(frame2,cv2.COLOR_BGR2YUV)
    histg = np.zeros([1000, 256])
    hist = cv2.calcHist([yuv], [1], None, [240], [16,240])
    #m = max(hist)
    hist = hist * 20000 / frame2.size
    for i in range(240):
        n = int(hist[i])
        cv2.line(histg, (i, 1000), (i, 1000-n), [255,255,255])

    flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)  # flow est une flow optique qui a deux channels
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme    amax(mag): le maximum dans le matrix mag

    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    result = np.vstack((frame2,bgr)) # combiner frame2 et bgr à l'orientation verticale
    
    #cv2.imshow('Image et Champ de vitesses (Farneback)',result.astype(np.uint8))
    cv2.namedWindow('histogramme 1d', cv2.WINDOW_NORMAL)
    cv2.imshow('Image et Champ de vitesses (Farneback)',frame2)
    cv2.imshow('histogramme 1d', histg)

    if index == 122:
        cv2.imwrite('histogramme_u_%04d.png'%index,frame2)
    if index == 282:
        cv2.imwrite('histogramme_u_%04d.png'%index,frame2)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
        cv2.imwrite('histogramme_%04d.png'%index,hist)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 
cap.release()
cv2.destroyAllWindows()
'''

'''
from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('../TP2_Videos_Codes/histogramme_u_0122.png')
img2 = cv2.imread('../TP2_Videos_Codes/histogramme_u_0282.png')

yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

hist = cv2.calcHist([yuv], [1], None, [240], [0, 240])
hist = hist/yuv.size

hist2 = cv2.calcHist([yuv2], [1], None, [240], [0, 240])
hist2 = hist2/yuv2.size

plt.figure()
plt.subplot(121)
plt.title("Histogramme corresspendant à la probabilité joint u")
plt.xlabel("valeur de u channel")
plt.ylabel("probabilité")
plt.plot(hist)
plt.xlim([0, 240])

plt.subplot(122)
plt.title("Histogramme corresspendant à la probabilité joint u")
plt.xlabel("valeur de u channel")
plt.ylabel("probabilité")
plt.plot(hist2)
plt.xlim([0, 240])
plt.show()
'''



