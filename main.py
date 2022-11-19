import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog
root = Tk()
root.withdraw()
filename = filedialog.askopenfilename(parent=root)

def ResizeImage(img):
    scale = 35
    width = int(img.shape[1] * scale / 75)
    height = int(img.shape[0] * scale / 75)
    desiredSize = (width, height)
    out = cv2.resize(img, desiredSize)
    return out


while (1):
    # Read image
    image_path = filename
    imageFrame = cv2.imread('fruits.jpeg')
    imageFrame = ResizeImage(imageFrame)

    # Convert from RGB to HSV
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set the red HSV range for lower boundary
    red_lower1 = np.array([0, 100, 200])
    red_upper1 = np.array([10, 255, 255])

    # Set the red HSV range for upper boundary
    red_lower2 = np.array([160, 100, 20])
    red_upper2 = np.array([179, 255, 255])

    # Combine red masks together
    lower_red_mask = cv2.inRange(hsvFrame, red_lower1, red_upper1)
    upper_red_mask = cv2.inRange(hsvFrame, red_lower2, red_upper2)
    full_red_mask = lower_red_mask + upper_red_mask

    # Apply mask and text
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=full_red_mask)


    # Set the HSV range for green
    green_lower = np.array([25, 52, 72])
    green_upper = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Apply mask and text
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)


    # Set the HSV range for blue
    blue_lower = np.array([94, 80, 2])
    blue_upper = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Apply mask and text
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

    # Set the HSV range for yellow
    yellow_lower = np.array([8, 75, 75])
    yellow_upper = np.array([20, 255, 255])
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Apply mask and text
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask=yellow_mask)


    # Create contour for red colour
    contours, hierarchy = cv2.findContours(full_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    red_image_copy = imageFrame.copy()
    cv2.drawContours(red_image_copy, contours, -1, (0, 0, 255), 5, cv2.LINE_AA)

    # Create contour for green colour
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    green_image_copy = imageFrame.copy()
    cv2.drawContours(green_image_copy, contours, -1, (0, 255, 0), 5, cv2.LINE_AA)

    # Create contour for blue colour
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    blue_image_copy = imageFrame.copy()
    cv2.drawContours(blue_image_copy, contours, -1, (255, 0, 0), 5, cv2.LINE_AA)

    # Create contour for yellow colour
    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    yellow_image_copy = imageFrame.copy()
    cv2.drawContours(yellow_image_copy, contours, -1, (0, 255, 255), 5, cv2.LINE_AA)

    # Blend all image together for the Outline
    blend1 = cv2.addWeighted(red_image_copy, 0.5, green_image_copy, 0.5, 0)
    blend2 = cv2.addWeighted(blue_image_copy, 0.5, blend1, 0.5, 0)
    blend3 = cv2.addWeighted(yellow_image_copy, 0.5, blend2, 0.5, 0)

    # Add text to the final image
    blend3 = cv2.putText(blend3, 'Outline', (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (255, 255, 255), 2, cv2.LINE_AA)

    # Combine the images by concatenating
    ogandoutline_image = np.concatenate((imageFrame, blend3), axis=1)
    splitcoloured_Image = np.concatenate((res_red, res_green, res_blue, res_yellow), axis=1)
    cv2.imshow('Color segmentation', splitcoloured_Image)

    # Press Q to quit the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
