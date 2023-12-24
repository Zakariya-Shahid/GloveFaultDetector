import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

root = tk.Tk()

root.title('Defects Detector')
root.resizable(False, False)
root.geometry('300x300')


def clear():
    listp = root.pack_slaves()
    for l in listp:
        if "frame" in str(l):
            l.destroy()
    select_file()


def select_file():
    filetypes = (
        ('jpg', '*.jpg'),
        ('jpeg', '*.jpeg'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    # Define the different types of defects
    defect_types = {
        0: 'Hole',
        1: 'Oil Stain',
        2: 'Thread Hanging',
        3: 'Thread Slippage',
        4: 'Needle Cut',
        5: 'Needle Hole',
        6: 'Glue Stain',
        7: 'Burn',
        8: 'Seam Opening'
    }

    # Load the image of the glove
    img = cv2.imread(filename)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the parameters for the blob detector
    params = cv2.SimpleBlobDetector_Params()

    # Set the minimum and maximum size of the blobs to be detected
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 20000

    # Create the blob detector object
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect the blobs in the image
    keypoints = detector.detect(gray)

    # Draw the keypoints on the original image
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Define the region of interest for each defect type
    rois = [
        # Hole
        (200, 200, 300, 300),
        # Oil Stain
        (500, 200, 600, 300),
        # Thread Hanging
        (800, 200, 900, 300),
        # Thread Slippage
        (200, 500, 300, 600),
        # Needle Cut
        (500, 500, 600, 600),
        # Needle Hole
        (800, 500, 900, 600),
        # Glue Stain
        (200, 800, 300, 900),
        # Burn
        (500, 800, 600, 900),
        # Seam Opening
        (800, 800, 900, 900)
    ]

    # Loop through the ROIs and check for defects
    defects_found = []
    for i, roi in enumerate(rois):
        # Extract the ROI from the image
        x1, y1, x2, y2 = roi
        roi_img = gray[y1:y2, x1:x2]

        # Threshold the ROI to isolate defects
        _, thresh = cv2.threshold(roi_img, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found, there is a defect
        if len(contours) > 0:
            defects_found.append(defect_types[i])
            print('Defect found: {}'.format(defect_types[i]))
            # Draw a square on the defect area
            cv2.rectangle(img_with_keypoints, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

    # Print the defects found, if any
    if len(defects_found) > 0:
        print('Defects found:')
        for defect in defects_found:
            frame1 = tk.Frame(root, background='#dfdfdf')
            frame1.pack(fill=tk.X)

            tk.Label(frame1, text=defect, background="#dfdfdf").pack(side=tk.LEFT, padx=60)
    else:
        print('No defects found.')

    # Show the image with the keypoints and defect areas
    cv2.imshow('Keypoints and Defect Areas', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


select_file()
