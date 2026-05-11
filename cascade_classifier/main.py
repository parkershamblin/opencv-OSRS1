import cv2 as cv
import numpy as np
import os
from pathlib import Path
from time import time, sleep
from windowcapture import WindowCapture
from vision import Vision
import pyautogui
from threading import Thread, Lock
import keyboard

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

REPO_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_DIR / 'training'
DEFAULT_COLLECT_DATASET = 'fixed_zoom_v1'


def _env_flag(name):
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


# initialize the WindowCapture class
# Prefers RuneLite* and falls back to the official OSRS client.
# Works on multi-monitor setups (including negative coordinates on secondary monitors)
wincap = WindowCapture(capture_from_screen=_env_flag('OSRS_CAPTURE_FROM_SCREEN'))

def _discover_cascade_paths():
    paths = [path for path in Path('.').glob('cascade*/cascade.xml') if path.is_file()]
    return [
        path.as_posix()
        for path in sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)
    ]


def _select_cascade_paths():
    raw_paths = os.environ.get('COW_CASCADE_PATHS') or os.environ.get('COW_CASCADE_PATH')
    if raw_paths:
        return [path.strip() for path in raw_paths.split(os.pathsep) if path.strip()]

    paths = _discover_cascade_paths()
    if not paths:
        raise RuntimeError('No cascade.xml files found under cascade*/ folders.')

    if len(paths) == 1:
        print(f'Using only cascade model: {paths[0]}')
        return paths

    newest_path = max(paths, key=lambda path: Path(path).stat().st_mtime)
    default_index = paths.index(newest_path)

    print('Available cascade models:')
    for index, path in enumerate(paths, 1):
        default_marker = ' (newest/default)' if index - 1 == default_index else ''
        print(f'  {index}. {path}{default_marker}')
    print('Type a number, "all", or press Enter for the newest model.')

    while True:
        choice = input(f'Select cascade [1-{len(paths)}]: ').strip().lower()
        if choice == '':
            return [paths[default_index]]
        if choice == 'all':
            return paths
        try:
            selected_index = int(choice) - 1
        except ValueError:
            selected_index = -1

        if 0 <= selected_index < len(paths):
            return [paths[selected_index]]

        print('Please enter a valid number, "all", or press Enter.')


def _load_cascades(paths):
    cascades = []
    for path in paths:
        cascade = cv.CascadeClassifier(path)
        if cascade.empty():
            raise RuntimeError(f'Failed to load cascade classifier: {path}')
        cascades.append((path, cascade))
    return cascades


def _ensure_dataset_dirs(dataset_dir):
    (dataset_dir / 'positive').mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'negative').mkdir(parents=True, exist_ok=True)


def _training_dataset_options():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    default_dir = TRAINING_DIR / DEFAULT_COLLECT_DATASET
    _ensure_dataset_dirs(default_dir)

    datasets = []
    for dataset_dir in sorted(path for path in TRAINING_DIR.iterdir() if path.is_dir()):
        _ensure_dataset_dirs(dataset_dir)
        datasets.append(dataset_dir)

    return datasets


def _dataset_sample_counts(dataset_dir):
    image_extensions = {'.bmp', '.jpeg', '.jpg', '.png'}
    positive_count = len([
        path for path in (dataset_dir / 'positive').iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ])
    negative_count = len([
        path for path in (dataset_dir / 'negative').iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ])
    return positive_count, negative_count


def _select_training_dataset():
    datasets = _training_dataset_options()
    default_dir = TRAINING_DIR / DEFAULT_COLLECT_DATASET
    default_index = datasets.index(default_dir) if default_dir in datasets else 0

    print('Available training datasets:')
    for index, dataset_dir in enumerate(datasets, 1):
        positive_count, negative_count = _dataset_sample_counts(dataset_dir)
        default_marker = ' (default)' if index - 1 == default_index else ''
        print(
            f'  {index}. {dataset_dir.name} '
            f'({positive_count} positive, {negative_count} negative){default_marker}'
        )
    print('Type a number, a new dataset name, or press Enter for the default.')

    while True:
        choice = input(f'Select training dataset [1-{len(datasets)}]: ').strip()
        if choice == '':
            dataset_dir = datasets[default_index]
            break
        try:
            selected_index = int(choice) - 1
        except ValueError:
            selected_index = -1

        if 0 <= selected_index < len(datasets):
            dataset_dir = datasets[selected_index]
            break
        if choice:
            safe_name = choice.replace(' ', '_')
            dataset_dir = TRAINING_DIR / safe_name
            _ensure_dataset_dirs(dataset_dir)
            break

        print('Please enter a valid number, a dataset name, or press Enter.')

    print(f'Collecting samples into: {dataset_dir}')
    return dataset_dir


# load an empty Vision class
vision_cow = Vision()

DETECTION_SCALE_FACTOR = 1.05
DETECTION_MIN_NEIGHBORS = 8
DETECTION_MIN_SIZE = (40, 32)
DETECTION_MAX_SIZE = (260, 220)
DETECTION_ASPECT_RATIO_RANGE = (0.75, 1.6)
DETECTION_OVERLAP_THRESHOLD = 0.35
MATCHES_WINDOW_NAME = 'Matches'
PREVIEW_MAX_WIDTH = 960
PREVIEW_MAX_HEIGHT = 540
PREVIEW_SCREEN_FRACTION = 0.55
PREVIEW_WINDOW_MARGIN = 24
preview_window_initialized = False

# this global variable is used to notify the main loop of when the bot
# actions have completed
is_bot_in_action = False

# Fullscreen transition detection and recovery
consecutive_capture_failures = 0
MAX_CAPTURE_FAILURES_BEFORE_REINIT = 5  # Reinit after 5 failed captures
last_good_screenshot = None

# Ask user which mode to run in: collect samples or run inference
mode = None
while True:
    choice = input("Select mode: [c]ollect samples (press f/g to save) or [r]un inference (bot active): ").strip().lower()
    if choice in ('c', 'collect'):
        mode = 'collect'
        break
    if choice in ('r', 'run', 'inference'):
        mode = 'run'
        break
    print('Please enter "c" or "r"')

print(f"Mode selected: {mode}")

positive_sample_dir = Path('positive')
negative_sample_dir = Path('negative')
if mode == 'collect':
    collect_dataset_dir = _select_training_dataset()
    positive_sample_dir = collect_dataset_dir / 'positive'
    negative_sample_dir = collect_dataset_dir / 'negative'

cascade_cows = []
if mode == 'run':
    cascade_cows = _load_cascades(_select_cascade_paths())
    print('Loaded cascade classifiers:')
    for cascade_path, _ in cascade_cows:
        print(f'  {cascade_path}')
# shared screenshot for hotkey handlers
latest_screenshot = None
screenshot_lock = Lock()

# quit flag for global hotkey
quit_requested = False


def _preview_size_for_frame(frame):
    frame_height, frame_width = frame.shape[:2]
    try:
        screen_width, screen_height = pyautogui.size()
        max_width = min(PREVIEW_MAX_WIDTH, int(screen_width * PREVIEW_SCREEN_FRACTION))
        max_height = min(PREVIEW_MAX_HEIGHT, int(screen_height * PREVIEW_SCREEN_FRACTION))
    except Exception:
        max_width = PREVIEW_MAX_WIDTH
        max_height = PREVIEW_MAX_HEIGHT

    scale = min(max_width / frame_width, max_height / frame_height, 1.0)
    return max(1, int(frame_width * scale)), max(1, int(frame_height * scale))


def _preview_window_position(preview_width, preview_height):
    try:
        screen_width, screen_height = pyautogui.size()
        game_left = max(0, wincap.offset_x)
        game_top = max(0, wincap.offset_y)
        game_right = game_left + wincap.w
        game_bottom = game_top + wincap.h

        candidates = [
            (game_right + PREVIEW_WINDOW_MARGIN, game_top),
            (game_left - preview_width - PREVIEW_WINDOW_MARGIN, game_top),
            (game_left, game_bottom + PREVIEW_WINDOW_MARGIN),
            (
                screen_width - preview_width - PREVIEW_WINDOW_MARGIN,
                PREVIEW_WINDOW_MARGIN,
            ),
        ]

        for x, y in candidates:
            if (
                x >= PREVIEW_WINDOW_MARGIN
                and y >= PREVIEW_WINDOW_MARGIN
                and x + preview_width + PREVIEW_WINDOW_MARGIN <= screen_width
                and y + preview_height + PREVIEW_WINDOW_MARGIN <= screen_height
            ):
                return int(x), int(y)

        return (
            max(PREVIEW_WINDOW_MARGIN, screen_width - preview_width - PREVIEW_WINDOW_MARGIN),
            PREVIEW_WINDOW_MARGIN,
        )
    except Exception:
        return PREVIEW_WINDOW_MARGIN, PREVIEW_WINDOW_MARGIN


def _initialize_preview_window(frame):
    global preview_window_initialized
    if preview_window_initialized:
        return

    flags = cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO
    if hasattr(cv, 'WINDOW_GUI_NORMAL'):
        flags |= cv.WINDOW_GUI_NORMAL
    cv.namedWindow(MATCHES_WINDOW_NAME, flags)

    preview_width, preview_height = _preview_size_for_frame(frame)
    cv.resizeWindow(MATCHES_WINDOW_NAME, preview_width, preview_height)
    x, y = _preview_window_position(preview_width, preview_height)
    cv.moveWindow(MATCHES_WINDOW_NAME, x, y)
    preview_window_initialized = True


def show_preview(frame):
    _initialize_preview_window(frame)
    preview_width, preview_height = _preview_size_for_frame(frame)
    if frame.shape[1] != preview_width or frame.shape[0] != preview_height:
        frame = cv.resize(frame, (preview_width, preview_height), interpolation=cv.INTER_AREA)
    cv.imshow(MATCHES_WINDOW_NAME, frame)


# helper to save a screenshot safely from hotkey threads
def _save_sample(img, folder):
    if img is None:
        print('No image available to save yet')
        return
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = time()
    filename = folder / f"{timestamp:.6f}.jpg"
    # write in a separate thread to avoid blocking keyboard handler
    def _writer(im, fname):
        cv.imwrite(str(fname), im)
        print(f'Saved {fname}')
    t = Thread(target=_writer, args=(img, filename))
    t.daemon = True
    t.start()

# hotkey callbacks (only registered in collect mode)
def _hotkey_save_positive():
    with screenshot_lock:
        img = None if latest_screenshot is None else latest_screenshot.copy()
    _save_sample(img, positive_sample_dir)

def _hotkey_save_negative():
    with screenshot_lock:
        img = None if latest_screenshot is None else latest_screenshot.copy()
    _save_sample(img, negative_sample_dir)

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
        
        # pyautogui.moveTo(x=screen_x, y=screen_y)
        # pyautogui.click()
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
            show_preview(display_frame)
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
            gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
            rectangles = []
            for _, cascade_cow in cascade_cows:
                detections = cascade_cow.detectMultiScale(
                    gray,
                    scaleFactor=DETECTION_SCALE_FACTOR,
                    minNeighbors=DETECTION_MIN_NEIGHBORS,
                    minSize=DETECTION_MIN_SIZE,
                    maxSize=DETECTION_MAX_SIZE,
                )
                rectangles.extend(detections.tolist() if len(detections) else [])
            rectangles = vision_cow.filter_rectangles(
                rectangles,
                min_size=DETECTION_MIN_SIZE,
                aspect_ratio_range=DETECTION_ASPECT_RATIO_RANGE,
                overlap_threshold=DETECTION_OVERLAP_THRESHOLD,
            )
            # draw the detection results onto the original image
            detection_image = vision_cow.draw_rectangles(screenshot.copy(), rectangles)
            cv.putText(
                detection_image,
                f'detections: {len(rectangles)}',
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            
            # Add status overlay if capture is failing
            if consecutive_capture_failures > 0:
                cv.putText(detection_image, f'CAPTURE FAILING ({consecutive_capture_failures})', 
                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # display the images
            show_preview(detection_image)
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
