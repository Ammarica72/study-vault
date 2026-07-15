"""
============================================================
Video Frame Extraction Tool
============================================================

Purpose
-------
This program extracts every frame from a video and saves each
frame as an individual image. Each saved image includes both
its frame number and the time (in milliseconds) at which that
frame appeared in the video.

Saving the timestamp in the filename makes it easy to determine
when an event occurred during the video. This can be useful for
motion analysis, physics experiments, or any activity where the
timing of events is important.

------------------------------------------------------------
Input
------------------------------------------------------------

The program expects:

• A video file (specified by video_path).

For example:
    VID_20260527_121243253.mp4

------------------------------------------------------------
Output
------------------------------------------------------------

The program creates an output folder (specified by output_folder)
containing one image for every frame in the video.

Each image is named using the format:

    frame_0000_time_0ms.jpg
    frame_0001_time_33ms.jpg
    frame_0002_time_67ms.jpg
    ...

where:
    • frame_0000 is the frame number.
    • time_0ms is the timestamp of that frame in milliseconds.

The original video is NOT modified.

------------------------------------------------------------
Libraries Used
------------------------------------------------------------

OpenCV (cv2)
    Opens the video, reads each frame, and saves the images.

os
    Creates folders and works with file paths.

------------------------------------------------------------
Workflow
------------------------------------------------------------

1. Open the video file.
2. Create the output folder if it does not already exist.
3. Read the video one frame at a time.
4. Record the timestamp of each frame.
5. Save every frame as an image with its timestamp included in
   the filename.
6. Continue until the end of the video is reached.

"""


import cv2
import os

def extract_frames_with_timestamps(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    
    while True:
        # 1. Get the timestamp of the CURRENT frame in milliseconds
        timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
        
        # 2. Read the frame
        success, frame = video.read()
        
        if not success:
            break
        
        # 3. Round the millisecond for cleaner filenames (e.g., 123.45ms -> 123ms)
        timestamp_rounded = round(timestamp_ms)
        
        # 4. Save the frame with the timestamp in the filename
        frame_name = os.path.join(output_folder, f"frame_{frame_count:04d}_time_{timestamp_rounded}ms.jpg")
        cv2.imwrite(frame_name, frame)
        
        print(f"Saved frame {frame_count} at {timestamp_rounded} ms")
        frame_count += 1
    
    video.release()
    print(f"\nDone! Extracted {frame_count} frames.")

# --- Run the function ---
# The name of the input mp4 needs to change to the name of the input you have
extract_frames_with_timestamps('VID_20260527_121243253.mp4', 'ball_drop')
