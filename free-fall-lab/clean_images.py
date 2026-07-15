
"""
============================================================
Image Blur Tool for Batch Image Processing
============================================================

Purpose
-------
This program allows you to blur the same region in every image
within a folder. It is useful when you need to hide or anonymize
an object (for example, a person's face, a logo, or identifying
information) across a sequence of images.

Instead of manually editing every image, you only need to:

1. Choose one sample image.
2. Draw a rectangle around the area you want to blur.
3. Adjust the blur strength until you are happy with the result.
4. The program automatically applies the same blur to every image
   in the input folder.

------------------------------------------------------------
Input
------------------------------------------------------------

The program expects:

• A folder containing all of the images you want to process
  (specified by INPUT_FOLDER).

• One sample image from that folder
  (specified by SAMPLE_IMAGE).
  This image is only used to choose the blur location and
  preview the blur strength.

------------------------------------------------------------
Output
------------------------------------------------------------

The program creates a new folder (OUTPUT_FOLDER) containing
copies of every image with the selected region blurred.

The original images are NOT modified.

------------------------------------------------------------
Libraries Used
------------------------------------------------------------

OpenCV (cv2)
    Reads, displays, edits, and saves images.

os
    Works with folders and file paths.

------------------------------------------------------------
Workflow
------------------------------------------------------------

1. Load the sample image.
2. Resize it for easier viewing.
3. Draw a rectangle around the object to blur.
4. Preview and adjust the blur strength.
5. Apply the same blur to every image in the input folder.
6. Save the processed images into the output folder.

"""



# Import the OpenCV library (used for reading, displaying, and editing images)
import cv2

# Import the os library (used for working with folders and file names)
import os

# ============================================================
# USER SETTINGS
# Change these values if your image folders or sample image
# have different names.
# ============================================================

# Folder containing the original images
INPUT_FOLDER = 'ball_drop'

# Folder where the blurred images will be saved
OUTPUT_FOLDER = 'blurred_images'

# One example image that will be used to choose the blur area
SAMPLE_IMAGE = 'frame_0000_time_0ms.jpg'

# Height (in pixels) used for the preview image.
# The preview is made smaller so it fits comfortably on the screen.
TARGET_H = 600


# ============================================================
# CREATE OUTPUT FOLDER (if it doesn't already exist)
# ============================================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


# ============================================================
# LOAD THE SAMPLE IMAGE
# ============================================================

# Build the full file path to the sample image
img_path = os.path.join(INPUT_FOLDER, SAMPLE_IMAGE)

# Read the image into Python
img_original = cv2.imread(img_path)


# Continue only if the image was loaded successfully
if img_original is not None:

    # Get the original image height and width
    h_orig, w_orig = img_original.shape[:2]

    # Calculate how much the image should be resized
    ratio = TARGET_H / float(h_orig)

    # Resize the image for easier viewing
    img_preview = cv2.resize(
        img_original,
        (int(w_orig * ratio), TARGET_H)
    )

    # Variables used later in the program
    running = True
    final_roi = None       # ROI = Region Of Interest (the selected rectangle)
    final_k_size = None    # Final blur strength

    # ========================================================
    # STEP 1 - SELECT THE AREA TO BLUR
    # ========================================================

    while running:

        print("\n[ACTION] Draw a rectangle around the object you want to blur.")
        print("Press ENTER or SPACE after drawing to finish.")

        # Opens a window where the user draws a rectangle.
        roi = cv2.selectROI(
            "Drawing Window",
            img_preview,
            fromCenter=False
        )

        # Save the rectangle coordinates
        x, y, w, h = [int(v) for v in roi]

        # Close the drawing window
        cv2.destroyWindow("Drawing Window")

        # Ask if the selected rectangle looks correct
        ans = input("Coordinate acceptable? (y/n): ").lower()

        if ans == 'y':

            # Starting blur strength.
            # Larger numbers produce a stronger blur.
            blur_intensity = 51

            # ====================================================
            # STEP 2 - ADJUST THE BLUR STRENGTH
            # ====================================================

            while True:

                # Make a copy so the original preview isn't changed
                img_blurred = img_preview.copy()

                # Only blur if a rectangle was actually selected
                if w > 0 and h > 0:

                    # Extract only the selected region
                    roi_section = img_preview[y:y+h, x:x+w]

                    # Gaussian Blur requires an odd kernel size
                    if blur_intensity % 2 != 0:
                        k_size = blur_intensity
                    else:
                        k_size = blur_intensity + 1

                    # Apply the blur
                    img_blurred[y:y+h, x:x+w] = cv2.GaussianBlur(
                        roi_section,
                        (k_size, k_size),
                        0
                    )

                # Display the blurred preview
                win_name = "Blurred Preview"

                cv2.namedWindow(win_name)

                # Keep the preview window in front
                cv2.setWindowProperty(
                    win_name,
                    cv2.WND_PROP_TOPMOST,
                    1
                )

                cv2.imshow(win_name, img_blurred)
                cv2.waitKey(1)

                # Ask if the blur looks good
                ans_blur = input(
                    f"Is the blur (intensity {k_size}) acceptable? (y/n): "
                ).lower()

                if ans_blur == 'y':

                    print("Final answer accepted.")

                    # Save the rectangle and blur strength
                    final_roi = (x, y, w, h)
                    final_k_size = k_size

                    # Close all image windows
                    cv2.destroyAllWindows()

                    # Exit both loops
                    running = False
                    break

                else:
                    # Increase the blur and try again
                    blur_intensity += 40
                    print("Increasing blur and showing another preview...")

                    cv2.destroyWindow(win_name)

        else:
            # User wants to redraw the rectangle
            print("Restarting drawing process...")


    # ========================================================
    # STEP 3 - APPLY THE SAME BLUR TO EVERY IMAGE
    # ========================================================

    if final_roi and final_k_size:

        print("\n--- STARTING BATCH PROCESS ---")

        # Accept only common image file types
        valid_exts = (
            '.jpg',
            '.jpeg',
            '.png',
            '.bmp'
        )

        # Get every image in the input folder
        files = [
            f for f in os.listdir(INPUT_FOLDER)
            if f.lower().endswith(valid_exts)
        ]

        # Unpack the saved rectangle coordinates
        rx, ry, rw, rh = final_roi

        # Process each image one at a time
        for filename in files:

            full_path = os.path.join(INPUT_FOLDER, filename)

            # Load the full-size image
            frame = cv2.imread(full_path)

            # Skip files that cannot be opened
            if frame is None:
                continue

            # ----------------------------------------------------
            # IMPORTANT
            #
            # The rectangle was selected on a resized preview.
            # We must convert those coordinates back to the
            # original image size before applying the blur.
            # ----------------------------------------------------

            h_fm, w_fm = frame.shape[:2]

            scale = h_fm / float(TARGET_H)

            # Scale the rectangle coordinates
            sx, sy, sw, sh = [
                int(v * scale)
                for v in [rx, ry, rw, rh]
            ]

            if sw > 0 and sh > 0:

                # Extract the matching region
                roi_sub = frame[sy:sy+sh, sx:sx+sw]

                # Scale the blur amount to match the larger image
                scaled_k = int(final_k_size * scale)

                # Gaussian Blur requires an odd number
                if scaled_k % 2 == 0:
                    scaled_k += 1

                # Apply the blur
                frame[sy:sy+sh, sx:sx+sw] = cv2.GaussianBlur(
                    roi_sub,
                    (scaled_k, scaled_k),
                    0
                )

            # Save the edited image into the output folder
            cv2.imwrite(
                os.path.join(OUTPUT_FOLDER, filename),
                frame
            )

            print(f"Saved: {filename}")

else:
    # The sample image could not be found or loaded
    print(f"Error: Could not load {img_path}")


# ============================================================
# FINISHED
# ============================================================

print("\nAll tasks complete!")
print("Open the 'blurred_images' folder to see the results.")
