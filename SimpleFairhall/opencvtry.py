import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
from paintapp import Window
import sys

class cvWriter:
    # Define the codec and create VideoWriter object

    def __init__(self, filePathName, frameDim, frate=24., codec='MP42'):
        # For whatever reason OPENCV needs dimensions in the opposite order
        frameDimT = (frameDim[1], frameDim[0])

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._out = cv2.VideoWriter(filePathName, fourcc, float(frate), frameDimT, isColor=False)

    # Just necessary to use 'with' command
    def __enter__(self):
        return self

    # Destructor for the 'with' command
    # Release everything if job is finished
    def __exit__(self, exc_type, exc_value, traceback):
        self._out.release()
        # cv2.destroyAllWindows()

    # Write matrix frame to file
    def write(self, mat):
        self._out.write(mat.astype(np.uint8))




FPS = 24
seconds = 10
width = 1280
height =720
filePathName = "./noise_image.mp4"

if False:
    frameDim = [height,width]

    # Time a frame stays
    time_step = 1


    with cvWriter(filePathName, (height, width)) as vidwriter:
        for _ in range(np.int(seconds / (time_step))):
            frame0 = np.random.randint(0, 255,
                                      (height, width,4),
                                      dtype=np.uint8)
            frame = np.zeros((height, width, 4))
            frame[:,:,0] = 255 * np.ones((height, width))
            frame[:,:,3] = 255 * np.ones((height, width))
            for i in range(time_step * FPS):
                vidwriter.write(frame)




frame_img = np.load('frame_img.npy')

width = frame_img.shape[1]
height =frame_img.shape[0]

time_step = 1
newfilePathName = "./painting_vid.mp4"

with cvWriter(newfilePathName, (height, width)) as vidwriter:
    for _ in range(np.int(seconds / (time_step))):
        for i in range(time_step * FPS):
            vidwriter.write(frame_img)
