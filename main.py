import cv2
import numpy as np
import pytesseract
from skimage.filters import threshold_local
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2
import easyocr
IMAGE_WIDTH = 70
IMAGE_HEIGHT = 70
SHOW_STEPS = False

# load trained data from images and classifications
npa_classifications = np.loadtxt("classifications.txt", np.float32)
npa_flattenedImages = np.loadtxt("flattened_images.txt", np.float32)
# create KNN object
k_nearest = cv2.ml.KNearest_create()
# train KNN object with training data
k_nearest.train(npa_flattenedImages, cv2.ml.ROW_SAMPLE, npa_classifications)

# load plate image
img = cv2.imread('dataset/9.jpg')

V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 29, offset=15, method='gaussian')
thresh = (V > T).astype("uint8") * 255
thresh = cv2.bitwise_not(thresh)

plate = imutils.resize(img, width=400)
thresh = imutils.resize(thresh, width=400)

cv2.imshow("thresh", thresh)

edged = cv2.Canny(thresh, 10, 100)

contours, hir = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# sort contours by their area size so we search in biggest shapres first
contours = sorted(contours, key=cv2.contourArea, reverse=True)

print(contours)

cv2.waitKey(0)
cv2.destroyAllWindows()