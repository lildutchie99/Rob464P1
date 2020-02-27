import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
#from skimage.transform import hough_ellipse, rescale
#from skimage.draw import ellipse_perimeter
from sklearn.cluster import KMeans
from skimage.io import imread
import numpy as np
import cv2

# Load picture, convert to grayscale and detect edges
image_rgb = imread('redtag.jpg') #data.coffee()[0:220, 160:420]
#image_rgb = rescale(image_rgb, 0.2, anti_aliasing=True, multichannel=True)

img_hsv = color.rgb2hsv(image_rgb)
hueVals = img_hsv[:, :, 0]
satVals = img_hsv[:, :, 1]
img_hsv[hueVals < .9] = [0, 0, 0]
img_hsv[satVals < 0.6] = [0, 0, 0]
image_rgb = color.hsv2rgb(img_hsv)

image_gray = color.rgb2gray(image_rgb)
print('doing edge detection')
edges = canny(image_gray)
#edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

#e2 = np.zeros((edges.shape[0], edges.shape[1], 3))
#e2[edges] = [1, 1, 1]
#plt.imshow(e2)
#plt.show()

im2, contours, hierarchy = cv2.findContours(edges.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


#print('finding ellipses')
#edges = rescale(edges, 0.5, anti_aliasing=True)
#result = hough_ellipse(edges, accuracy=1, threshold=250, min_size=10, max_size=500)
#result = hough_ellipse(edges)
#result.sort(order='accumulator')

thresh = 5 #maximum sum-of-squares difference between x, y, a, b between two ellipses that indicates they're separate

refined = {}
for el in result:
    print(el)
    yc, xc, a, b = list(el)[1:5] #a = major axis
    r = (a + b)/2 #consider average axis length

    #if len(refined)==0 or min([abs(sum(features - x)**2) for x in refined]) > thresh:
    coord = (int(xc), int(yc))
    if len(refined) > 0:
        dist, key = min([(((key[0] - coord[0])**2 + (key[1] - coord[1])**2), key) for key in refined])
    else:
        dist = np.inf
        key = coord
    if(dist > thresh): #consider this a new cluster 
        refined[coord] = [r]
    else: #consider this a preexisting cluster
        refined[key].append(r)

for key in refined:
    print("ellipses at ", key, ": ", refined[key])
    if len(refined[key]) < 6:
        continue
    kmeans = KMeans(n_clusters=6, random_state=0).fit(np.array(refined[key]).reshape(-1, 1))
    print("Labels: ", kmeans.labels_)
    print("Clusters: ", kmeans.cluster_centers_)

#orientation = best[5]

# Draw the ellipse on the original image
#cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
#image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
#edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()