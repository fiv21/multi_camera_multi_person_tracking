import cv2      # import the OpenCV library                     
import numpy as np  # import the numpy library
import matplotlib.pyplot as plt                                         
# provide points from image 1

im_src = cv2.imread('topview.png')
cv2.imshow('img_src', im_src)


pts_src = np.array([[900,524],[1550,313],[650,236],[0,447]])
# pts_src = np.array([[353,155], [558,223], [260,368],[55,300]])
# corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
pts_dst = np.array([[0, 0],[1000, 0],[1000, 1000],[0, 1000]])


plt.scatter(x=pts_src[:, 0], y=pts_src[:, 1])
plt.show()

plt.scatter(x=pts_dst[:, 0], y=pts_dst[:, 1])
plt.show()
# calculate matrix H
h, status = cv2.findHomography(pts_src, pts_dst)
 
# provide a point you wish to map from image 1 to image 2
a = np.array([[342,96],[636,27],[648,396],[455,594]], dtype='float32')
a = np.array([a])
 
# finally, get the mapping
imOut = cv2.perspectiveTransform(a, h)
imOut = np.array([imOut])
print(imOut)
x = []
y = []
for i in imOut:
        for j in i[0]:
            x.append(j[0])
            y.append(j[1])

plt.scatter(x,y)
plt.show()

# print(x)
# print(y)




# plt.show()