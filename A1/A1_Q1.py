import cv2
import numpy as np

def apply_invisible_cloak(frame, background):
    # convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define red color ranges and create masks
    lower_green1 = np.array([35, 60, 40], dtype=np.uint8)
    upper_green1 = np.array([85, 255, 255], dtype=np.uint8)

    # second range helps catch slightly different greens (lighting/shade differences)
    lower_green2 = np.array([25, 40, 30], dtype=np.uint8)
    upper_green2 = np.array([35, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)

    # combine masks and refine if needed
    mask = cv2.bitwise_or(mask1, mask2)
    # cv2.morphologyEx() applies morphological operations to improve a mask.
    # OPEN removes small noise (erosion then dilation).
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Segment out the green area using the mask (cloak area)
    cloak = cv2.bitwise_and(background, background, mask=mask)

    # create inverse mask and isolate cloak area
    inv_mask = cv2.bitwise_not(mask)
    rest = cv2.bitwise_and(frame, frame, mask=inv_mask)

    # combine background with current frame
    final_output = cv2.add(rest, cloak)

    return final_output


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("opened:", cap.isOpened())
# capture background (press 'b' to save it)
# or ignore this and use the first 2 seconds of the camera as background
while True:
    ret, background = cap.read()
    cv2.imshow("Background", background)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

while True:
    ret, frame = cap.read()
    output = apply_invisible_cloak(frame, background)
    cv2.imshow("Cloak Effect", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
