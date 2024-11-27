"""
Credits:
This project was highly inspired, and uses code snippets and concepts from Learn Code By Gaming's "OpenCV Object Detection in Games" YouTube Playlist.

Distinction:
My project is unique ffrom the formentioned tutorial in game, dataset, and implementation, but with that being said, their tutorial was still
fundamental in teaching me how to train my own cascade classifer model for object detection in RuneScape and I use several of their code snippets
throughout this project.
"""

import cv2 as cv
import numpy as np
import os
from time import time, sleep
from windowcapture import WindowCapture
from vision import Vision
import pyautogui
from threading import Thread

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
wincap = WindowCapture('Old School Runescape')

# load the trained model
cascade_cow = cv.CascadeClassifier('cascade/cascade.xml')
# load an empty Vision class
vision_cow = Vision(None)

# this global variable is used to notify the main loop of when the bot
# actions have completed
is_bot_in_action = False

# this function will be performed inside another thread
def bot_actions(rectangles):
    # take bot actions
    if len(rectangles) > 0:
        # just grab the first objects detection in the list and find the place
        # to click
        targets = vision_cow.get_click_points(rectangles)
        target = wincap.get_screen_position(targets[0])
        pyautogui.moveTo(x=target[0], y=target[1])
        pyautogui.click()
        # wait 10 seconds for killing cow to complete
        sleep(10)
        
    # let the main loop know when this process is completed
    global is_bot_in_action
    is_bot_in_action = False

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # use cascade classifier to do object detection
    rectangles = cascade_cow.detectMultiScale(screenshot)

    # draw the detection results onto the original image
    detection_image = vision_cow.draw_rectangles(screenshot, rectangles)

    # display the images
    cv.imshow('Matches', detection_image)
    
    # take bot actions
    #   run the function in a thread that's seperate from the main thread
    #   so that the code hree can continue while the bot performs its actions
    if not is_bot_in_action:
        is_bot_in_action = True
        t = Thread(target=bot_actions, args=(rectangles,))
        t.start()

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # press 'f' to save screenshot as a positive image, press 'd' to 
    # save as a negative image.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('f'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
    elif key == ord('d'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)

print('Done.')
