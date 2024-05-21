import cv2
import numpy as np


def toGrayAvg(im):
    r, g, b = im[:, :, 2], im[:, :, 1], im[:, :, 0]
    gray = (r + g + b) / 3
    return gray.astype(np.uint8)


def toGray(im):
    r, g, b = im[:, :, 2], im[:, :, 1], im[:, :, 0]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)


# Load the image
im = cv2.imread("./im1.jpg")

# Display the image
cv2.imshow("image", im)

# Convert the image to gray scale using average method
im_gray_avg = toGrayAvg(im)

cv2.imshow("gray_avg", im_gray_avg)

im_gray = toGray(im)
cv2.imshow("gray", im_gray)


while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
