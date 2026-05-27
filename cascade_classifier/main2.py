import pyautogui
import time
try:
    import keyboard  # pip install keyboard
except Exception:
    keyboard = None

# track whether a quit has been requested
quit_requested = False

def _hotkey_quit():
    global quit_requested
    quit_requested = True
    print("Quit hotkey pressed. Exiting...", flush=True)

# register global hotkey if possible
if keyboard:
    # bind Escape to quit; runs in background thread
    keyboard.add_hotkey('esc', _hotkey_quit)
else:
    print('keyboard module not available; press Ctrl+C to stop', flush=True)

try:
    while True:
        if quit_requested:
            break
        pyautogui.moveRel(1,0)
        time.sleep(0.1)
        pyautogui.moveRel(-1,0)
        time.sleep(0.1)
except KeyboardInterrupt:
    pass