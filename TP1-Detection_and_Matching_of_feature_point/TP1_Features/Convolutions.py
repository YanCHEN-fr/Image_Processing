import numpy as np
import cv2

from matplotlib import pyplot as plt

# Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('/Users/apple/Desktop/ENSTA/apprentissage/3A/ROB317/TP/TP1/TP1_Features/Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

# Méthode directe
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
    img2[y,x] = min(max(val,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

plt.subplot(331)
plt.imshow(img, cmap = 'gray')
plt.title('original')

plt.subplot(332)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Méthode Directe')

# Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

plt.subplot(333)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')

# Derive en x
kernel2 = np.array([[-1, 0, 1],
				   [-2, 0, 2],
				   [-1, 0, 1]])
img4 = cv2.filter2D(img, -1, kernel2)

# Derive en y
kernel3 = np.array([[-1, -2, -1],
				    [0, 0, 0],
				    [1, 2, 1]])
img5 = cv2.filter2D(img, -1, kernel3)

# Norme de gradient
img6 = np.sqrt(img4*img4 + img5*img5)

# Arg gradient 
arg_gradient = np.arctan(img5/img4)

plt.subplot(334)
plt.imshow(img4,cmap = 'gray',vmin = -255.0,vmax = 255.0)
plt.title('Convolution - gradient en x')

plt.subplot(335)
plt.imshow(img5,cmap = 'gray',vmin = -255.0,vmax = 255.0)
plt.title('Convolution - gradient en y')

plt.subplot(336)
plt.imshow(img6,cmap = 'gray',vmin = -255.0,vmax = 255.0)
plt.title('Convolution - Norme')

plt.subplot(337)
plt.imshow(arg_gradient,cmap = 'gray',vmin = -255.0,vmax = 255.0)
plt.title('Convolution - arg_gradient')
plt.show()








