# https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
# %%
import cv2 
import matplotlib.pyplot as plt

#reading image
input_file = 'NDParis-reduce.jpg'
path = "/home/tttienthinh/Documents/Programmation/VaccineArm/image/visage/"
input_file = f"{path}photo-r-1.jpg"
img1 = cv2.imread(input_file)  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1)
plt.imshow(img_1)
plt.savefig(f"{path}point1.jpg")
# %%
import cv2 
import matplotlib.pyplot as plt

path = "/home/tttienthinh/Documents/Programmation/VaccineArm/image/visage/"
# read images
img1 = cv2.imread(f"{path}photo-r-1.jpg")  
img2 = cv2.imread(f"{path}photo-r-1-rotated.jpg") 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

figure, ax = plt.subplots(1, 3, figsize=(16, 8))

ax[0].imshow(img1, cmap='gray')
ax[2].imshow(img2, cmap='gray')

#sift
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

len(keypoints_1), len(keypoints_2)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
ax[1].imshow(img3)
plt.savefig(f"{path}result2.jpg")
plt.show()
# %%
plt.show()
# %%
