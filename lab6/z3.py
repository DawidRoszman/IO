# Bird couter

import cv2
import os

# Load images from a direcotry
images = os.listdir("./bird_miniatures")

print(images)

for image in images:
    img = cv2.imread(f"./bird_miniatures/{image}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply birghness filter
    blurred_img = cv2.GaussianBlur(gray_img, (1, 1), 0)
    thresh = cv2.adaptiveThreshold(
        blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations (closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # brightness = 100
    # bright_img = cv2.addWeighted(blurred_img, 1, blurred_img, 0, brightness)
    _, thresh = cv2.threshold(
        closed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours based on area
    min_area = 2
    max_area = 5000
    contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Count the number of birds
    num_birds = len(contours)

    print(f"Number of birds in {image} is {num_birds}")
    # Draw bounding boxes around detected birds
    img_with_boxes = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow(f"{image}", img_with_boxes)
    cv2.imshow(f"Original {image}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
