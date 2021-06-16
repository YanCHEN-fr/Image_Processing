import numpy as np
import cv2

from matplotlib import pyplot as plt

def gaussian_kernel(kernel_size,sigma):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky)) 

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('/Users/apple/Desktop/ENSTA/apprentissage/3A/ROB317/TP/TP1/TP1_Features/Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
# Mettre ici le calcul de la fonction d'intérêt de Harris
alpha = 0.01
#kenelH = (1/9)*np.ones((3,3),np.uint8)
kenelH = gaussian_kernel(3, 3)
kenelIx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
Ix = cv2.filter2D(img, -1, kenelIx)
kenelIy = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
Iy = cv2.filter2D(img, -1, kenelIy)
H11 = cv2.filter2D(Ix*Ix, -1, kenelH)
H12 = cv2.filter2D(Ix*Iy, -1, kenelH)
H21 = cv2.filter2D(Iy*Ix, -1, kenelH)
H22 = cv2.filter2D(Iy*Iy, -1, kenelH)

Theta = H11*H22 - H12*H21 - alpha*(H11+H22)**2
#
#
# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)

#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0

#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
					 [0, 1, 0, 1, 0],
					 [0, 0, 1, 0, 0],
					 [0, 1, 0, 1, 0],
					 [1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('/Users/apple/Desktop/ENSTA/apprentissage/3A/ROB317/TP/TP1/TP1_Features/Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()


