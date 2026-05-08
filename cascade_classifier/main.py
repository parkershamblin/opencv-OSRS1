import cv2 as cv
import numpy as np
import os
from time import time, sleep
from windowcapture import WindowCapture
from vision import Vision
import pyautogui
from threading import Thread, Lock
import keyboard

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
# Prefers RuneLite* and falls back to the official OSRS client.
# Works on multi-monitor setups (including negative coordinates on secondary monitors)
wincap = WindowCapture()

# load the trained model
cascade_cow = cv.CascadeClassifier('cascade/cascade.xml')
# load an empty Vision class
vision_cow = Vision()

# this global variable is used to notify the main loop of when the bot
# actions have completed
is_bot_in_action = False

# Fullscreen transition detection and recovery
consecutive_capture_failures = 0
MAX_CAPTURE_FAILURES_BEFORE_REINIT = 5  # Reinit after 5 failed captures
last_good_screenshot = None

# Ensure sample folders exist
os.makedirs('positive', exist_ok=True)
os.makedirs('negative', exist_ok=True)

# Ask user which mode to run in: collect samples or run inference
mode = None
while True:
    choice = input("Select mode: [c]ollect samples (press f/d to save) or [r]un inference (bot active): ").strip().lower()
    if choice in ('c', 'collect'):
        mode = 'collect'
        break
    if choice in ('r', 'run', 'inference'):
        mode = 'run'
        break
    print('Please enter "c" or "r"')

print(f"Mode selected: {mode}")
# shared screenshot for hotkey handlers
latest_screenshot = None
screenshot_lock = Lock()

# quit flag for global hotkey
quit_requested = False

# helper to save a screenshot safely from hotkey threads
def _save_sample(img, folder):
    if img is None:
        print('No image available to save yet')
        return
    timestamp = time()
    filename = os.path.join(folder, f"{timestamp:.6f}.jpg")
    # write in a separate thread to avoid blocking keyboard handler
    def _writer(im, fname):
        cv.imwrite(fname, im)
        print(f'Saved {fname}')
    t = Thread(target=_writer, args=(img, filename))
    t.daemon = True
    t.start()

# hotkey callbacks (only registered in collect mode)
def _hotkey_save_positive():
    with screenshot_lock:
        img = None if latest_screenshot is None else latest_screenshot.copy()
    _save_sample(img, 'positive')

def _hotkey_save_negative():
    with screenshot_lock:
        img = None if latest_screenshot is None else latest_screenshot.copy()
    _save_sample(img, 'negative')

def _hotkey_quit():
    global quit_requested
    quit_requested = True
    print('Quit requested via hotkey...')
# this function will be performed inside another thread
def bot_actions(rectangles):
    """
    Perform bot actions on detected rectangles.
    
    Multi-monitor note: get_screen_position() correctly handles negative coordinates
    that occur when windows are on secondary monitors to the left of the primary monitor.
    """
    # take bot actions
    if len(rectangles) > 0:
        # just grab the first objects detection in the list and find the place
        # to click
        targets = vision_cow.get_click_points(rectangles)
        target = wincap.get_screen_position(targets[0])
        
        # Ensure coordinates are integers for pyautogui
        screen_x = int(target[0])
        screen_y = int(target[1])
        
        pyautogui.moveTo(x=screen_x, y=screen_y)
        pyautogui.click()
        # wait 10 seconds for killing cow to complete
        sleep(10)
        
    # let the main loop know when this process is completed
    global is_bot_in_action
    is_bot_in_action = False

loop_time = time()
while(True):

    # get an updated image of the game (handle capture failures gracefully)
    # Window capture may fail temporarily (e.g., when OSRS goes fullscreen)
    try:
        screenshot = wincap.get_screenshot()
        # successful capture - reset failure counter
        consecutive_capture_failures = 0
        last_good_screenshot = screenshot
    except Exception as e:
        consecutive_capture_failures += 1
        # print error on first failure or every 10 failures to avoid spam
        if consecutive_capture_failures == 1 or consecutive_capture_failures % 10 == 0:
            print(f'Capture failed #{consecutive_capture_failures} (will retry): {e}')
        
        # After MAX_CAPTURE_FAILURES_BEFORE_REINIT failures, try to reinitialize window
        # This handles fullscreen-to-windowed transitions
        if consecutive_capture_failures == MAX_CAPTURE_FAILURES_BEFORE_REINIT:
            print('Max failures reached; attempting to refresh/re-find window capture...')
            try:
                wincap.refresh_window_position()
                print('Window capture refreshed. Retrying capture.')
            except Exception as reinit_err:
                print(f'Reinit failed: {reinit_err}')
        
        # Use last good screenshot if available, otherwise create blank frame
        if last_good_screenshot is not None:
            screenshot = last_good_screenshot
        else:
            # Create blank frame as placeholder
            screenshot = np.zeros((480, 640, 3), dtype=np.uint8)
            cv.putText(screenshot, 'Waiting for window capture...', (50, 240), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        sleep(0.05)  # brief delay to avoid CPU spin

    # update shared screenshot for hotkey handlers
    with screenshot_lock:
        latest_screenshot = screenshot.copy()

    if mode == 'collect':
        # In collect mode we DO NOT draw detection boxes or perform bot actions.
        # This ensures training images saved via 'f' or 'd' are clean.
        try:
            # Add status overlay if capture is failing
            display_frame = screenshot.copy()
            if consecutive_capture_failures > 0:
                cv.putText(display_frame, f'CAPTURE FAILING ({consecutive_capture_failures})', 
                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow('Matches', display_frame)
        except Exception as e:
            print(f'Display failed: {e}')
        rectangles = []
        # register global hotkeys once per process (collect mode)
        if not globals().get('hotkeys_registered'):
            try:
                keyboard.add_hotkey('f', _hotkey_save_positive)
                keyboard.add_hotkey('g', _hotkey_save_negative)
                keyboard.add_hotkey('q', _hotkey_quit)
                globals()['hotkeys_registered'] = True
                print('Global hotkeys registered: f (positive), g (negative), q (quit)')
            except Exception as e:
                print('Failed to register global hotkeys:', e)
    else:
        # use cascade classifier to do object detection
        try:
            rectangles = cascade_cow.detectMultiScale(screenshot)
            # draw the detection results onto the original image
            detection_image = vision_cow.draw_rectangles(screenshot.copy(), rectangles)
            
            # Add status overlay if capture is failing
            if consecutive_capture_failures > 0:
                cv.putText(detection_image, f'CAPTURE FAILING ({consecutive_capture_failures})', 
                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # display the images
            cv.imshow('Matches', detection_image)
        except Exception as e:
            print(f'Detection or display failed: {e}')
            rectangles = []

        # take bot actions (only in run/inference mode)
        #   run the function in a thread that's seperate from the main thread
        #   so that the code here can continue while the bot performs its actions
        if not is_bot_in_action:
            is_bot_in_action = True
            t = Thread(target=bot_actions, args=(rectangles,))
            t.start()
        
        # register quit hotkey in run mode as well
        if not globals().get('quit_hotkey_registered'):
            try:
                keyboard.add_hotkey('q', _hotkey_quit)
                globals()['quit_hotkey_registered'] = True
                print('Quit hotkey registered: q (quit)')
            except Exception as e:
                print('Failed to register quit hotkey:', e)

    # debug the loop rate
    try:
        # print('FPS {:.1f}'.format(1 / (time() - loop_time)))
        pass
    except ZeroDivisionError:
        pass
    loop_time = time()

    # waits 1 ms every loop to process key presses
    try:
        key = cv.waitKey(1)
        if key == ord('q'):
            cv.destroyAllWindows()
            break
    except Exception:
        # waitKey can fail if window is not available; continue gracefully
        pass
    
    # check if quit was requested via global hotkey
    if quit_requested:
        cv.destroyAllWindows()
        break
    # local keys for saving are disabled when using global hotkeys; use global f/g in collect mode

print('Done.')
# cleanup hotkeys if registered
if globals().get('hotkeys_registered'):
    try:
        keyboard.clear_all_hotkeys()
    except Exception:
        pass
