# OpenCV Old School RuneScape Bot

A computer vision bot that uses cascade classifiers to detect and interact with objects in Old School RuneScape (OSRS). This project demonstrates real-time object detection, window capture, and game automation using OpenCV and Python.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running the Bot](#running-the-bot)
  - [Training Cascade Classifiers](#training-cascade-classifiers)
  - [Collecting Training Data](#collecting-training-data)
- [Configuration](#configuration)
- [Development](#development)
  - [Code Modules](#code-modules)
  - [Performance Metrics (KPIs)](#performance-metrics-kpis)
- [Troubleshooting](#troubleshooting)
- [Safety & Ethics](#safety--ethics)

---

## Overview

This project implements an automation bot for OSRS using:

- **OpenCV** for real-time object detection via cascade classifiers
- **Window Capture** to grab OSRS window screenshots
- **PyAutoGUI/PyDirectInput** to simulate mouse and keyboard inputs
- **Threading** for non-blocking bot actions

The bot detects targets (e.g., cows), calculates click points, and automates interactions within the game. It demonstrates a complete pipeline from training data collection → classifier training → real-time detection → bot automation.

### Key Features

- Real-time detection using Haar cascade classifiers
- Configurable detection parameters (scale factor, min neighbors, etc.)
- Threaded bot actions to prevent frame rate drops
- Window-relative click coordinates for reliable interactions
- Detection visualization (bounding boxes, click points)

---

## Prerequisites

### System Requirements

- **OS**: Windows 10 or later
- **Python**: 3.10 (as specified in `environment.yml`)
- **Game**: Old School RuneScape client installed and runnable

### Required Software

- Python 3.10
- pip or conda (for dependency management)
- Old School RuneScape client

---

## Setup & Installation

### Option 1: Using Conda (Recommended)

If you have Anaconda or Miniconda installed:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate opencv-runescape-env

# Verify installation
python -c "import cv2; print(cv2.__version__)"
```

### Option 2: Using pip

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2; print(cv2.__version__)"
```

### Post-Installation Verification

Run a quick test to ensure everything is working:

```bash
# From the cascade_classifier folder
cd cascade_classifier
python -c "
import cv2
import os
cascade = cv2.CascadeClassifier('cascade/cascade.xml')
print(f'Cascade loaded: {not cascade.empty()}')
print(f'Working directory: {os.getcwd()}')
"
```

### Multi-Monitor Setup

The bot now fully supports multi-monitor setups, including:

- **Secondary monitors with negative coordinates** (monitor to the left of primary)
- **DPI scaling** (high-DPI displays are handled by Windows automatically)
- **Window movement between monitors** (position is refreshed every 60 frames)

**No special configuration is needed** — the bot automatically handles:
1. Negative X/Y coordinates when OSRS is on a secondary monitor
2. Coordinate offset calculations for proper click positioning
3. Periodic position refresh to catch window movement between monitors

If you have a multi-monitor setup and the bot clicks in the wrong location:
- Ensure the OSRS window title is exactly correct (use `WindowCapture.list_window_names()` to verify)
- Move the OSRS window to ensure it's fully on one monitor (spanning multiple monitors may cause issues)
- Check that the detected click points appear correct in the visualization window

---

## Project Structure

```
opencv-OSRS1/
├── README.md                 # This file
├── requirements.txt          # pip dependencies
├── environment.yml           # Conda environment specification
├── docs/
│   ├── KPIs.md              # Performance metrics & goals
│   └── img/                 # Documentation images
└── cascade_classifier/       # Main bot code
    ├── main.py              # Bot main loop
    ├── vision.py            # Vision utilities (click points, drawing)
    ├── windowcapture.py     # Window capture & screenshot handling
    ├── cascadeutils.py      # Cascade training utilities
    ├── edgefilter.py        # Image preprocessing utilities
    ├── pos.txt              # List of positive training images
    ├── neg.txt              # List of negative training images
    ├── pos.vec              # Compiled positive samples for training
    ├── positive/            # Positive training images (targets)
    ├── negative/            # Negative training images (non-targets)
    └── cascade/             # Trained cascade classifier
        ├── cascade.xml      # Final cascade classifier model
        ├── params.xml       # Training parameters
        └── stage0-15.xml    # Individual cascade stages
```

---

## Usage

### Running the Bot

#### 1. Prepare the Game Window

```bash
# Make sure OSRS is running and the window title is exactly:
# "Old School RuneScape"
# (The bot looks for this window title to capture screenshots)
```

#### 2. Run the Main Bot

```bash
cd cascade_classifier
python main.py
```

When you run `python main.py` the program now prompts you to select a mode:

- **Collect samples** (`c`): run this mode when building a training dataset. The program will display the raw game screenshot (no bounding boxes) and allow you to press `f` to save a positive sample into `cascade_classifier/positive` or `d` to save a negative sample into `cascade_classifier/negative`.
- **Collect samples** (`c`): run this mode when building a training dataset. The program will display the raw game screenshot (no bounding boxes) and allow you to press `f` to save a positive sample into `cascade_classifier/positive` or `d` to save a negative sample into `cascade_classifier/negative`.
   - These keys are registered as global hotkeys, so you can keep focus in the OSRS game window while pressing `f`/`d` to capture samples.  - Press `q` (also a global hotkey) to quit the program at any time.   - On some Windows setups the global hotkey library may require elevated permissions; if hotkeys don't register, run the script from an elevated prompt or use the local `Matches` window keys as a fallback.
- **Run inference** (`r`): run this mode to perform detection and automated actions. The program will draw bounding boxes and execute bot actions; `f`/`d` are disabled in this mode to avoid contaminating training data with overlays.

Select the desired mode when prompted. This separation ensures training images are captured cleanly without detection overlays or automatic clicking.

#### 3. What the Bot Does

- Captures screenshots from the OSRS window every frame
- Detects target objects (e.g., cows) using the cascade classifier
- Draws bounding boxes and click points on detections
- Clicks on detected targets (if any are found)
- Waits 10 seconds between actions for the game to process
- **Gracefully handles fullscreen mode**: If OSRS switches to fullscreen, the bot will skip frames until capture resumes; the `Matches` window will not freeze

#### 4. Stop the Bot

Press `q` (global hotkey) to exit at any time. You can keep focus in the OSRS game window while pressing `q`; the bot will clean up resources and print "Done."

### Training Cascade Classifiers

#### Quick Start

The project includes a trained cascade classifier in `cascade/cascade.xml`. To retrain or improve it:

```bash
cd cascade_classifier

# 1. Generate positive samples vector file
# (See "Collecting Training Data" section first)
python cascadeutils.py --generate-vec

# 2. Train a new cascade
python cascadeutils.py --train

# 3. Test detection quality on held-out images
python cascadeutils.py --test-detection
```

#### Detailed Training Steps

##### Step 1: Collect Training Data

See "Collecting Training Data" section below.

##### Step 2: Generate pos.vec

This converts annotated positive images into a vector format OpenCV needs:

```bash
python cascadeutils.py --generate-vec
```

This command:
- Reads `pos.txt` (paths to positive images)
- Reads `neg.txt` (paths to negative images)
- Creates `pos.vec` with object annotations

**Important**: Ensure `pos.txt` and `neg.txt` contain valid file paths relative to the working directory.

##### Step 3: Train the Cascade

```bash
python cascadeutils.py --train
```

Training parameters (edit in `cascadeutils.py`):
- `numPos`: Number of positive samples (default: based on your data)
- `numNeg`: Number of negative samples (default: 2x positive count)
- `numStages`: Cascade stages (default: 15)
- `minHitRate`: Minimum detection rate per stage (default: 0.995)
- `maxFalseAlarmRate`: Maximum false positives (default: 0.5)

**Expected Duration**: 30 minutes to several hours depending on dataset size.

##### Step 4: Evaluate Detection Quality

```bash
python cascadeutils.py --test-detection
```

This measures:
- **Precision**: Correct detections / Total detections
- **Recall**: Correct detections / Total targets in images
- **Comparison**: Before/after retraining improvements

### Collecting Training Data

#### Manual Collection Workflow

1. **Run data capture mode** (if implemented):

```bash
python main.py --collect-positive  # Capture targets
python main.py --collect-negative  # Capture non-targets
```

2. **Manual Setup** (if no collection mode):

```bash
# Create positive/ folder
mkdir positive

# Take screenshots of your target object in various:
# - Angles and rotations
# - Lighting conditions
# - Zoom levels
# - Screen positions

# Save as: positive/cow_001.png, positive/cow_002.png, etc.
```

3. **Annotate Positive Images**

Use a tool like **OpenCV Annotation Tool** or **LabelImg** to annotate target objects:

```bash
# For each image in positive/, create a .txt annotation file
# Format: one rectangle per line
# xmin ymin xmax ymax

# Example: positive/cow_001.txt
# 150 200 250 300
```

Or use inline annotation:

```python
# Use vision.draw_rectangles() on your images during collection
```

4. **Organize Training Data**

```
cascade_classifier/
├── pos.txt          # List of positive image paths
├── neg.txt          # List of negative image paths
├── positive/        # Annotated positive images + .txt files
└── negative/        # Non-target images
```

**pos.txt format** (one per line):
```
positive/cow_001.png 1 150 200 100 100
positive/cow_002.png 1 200 150 90 110
...
```
(Format: path, count, x, y, width, height)

**neg.txt format** (one per line):
```
negative/background_001.png
negative/background_002.png
...
```

5. **Data Quality Checklist**

- [ ] At least 100 positive images (more is better)
- [ ] At least 200-300 negative images
- [ ] Annotations are accurate (no missed objects)
- [ ] Balanced lighting and angles
- [ ] No duplicate images
- [ ] All paths in pos.txt and neg.txt are valid

---

## Configuration

### Bot Parameters (in `main.py`)

```python
# Window capture (automatically handles multi-monitor setups)
wincap = WindowCapture('Old School RuneScape')

# Multi-monitor refresh interval (frames between position updates)
REFRESH_WINDOW_INTERVAL = 60  # Refresh every 60 frames (~2 seconds at 30 FPS)

# Cascade classifier
cascade_cow = cv.CascadeClassifier('cascade/cascade.xml')

# Detection parameters
rectangles = cascade_cow.detectMultiScale(
    gray_image,
    scaleFactor=1.1,      # Image pyramid scale (1.1 = 10% reduction)
    minNeighbors=10,      # Min neighbors to confirm detection
    minSize=(50, 50),     # Minimum detection size
    maxSize=(150, 150)    # Maximum detection size
)

# Bot action delay
sleep(10)  # Seconds to wait after clicking before next detection
```

### Detection Tuning

Adjust these parameters to improve detection:

| Parameter | Effect | Range | Notes |
|-----------|--------|-------|-------|
| `scaleFactor` | Faster but less accurate; slower but more thorough | 1.05–1.4 | Start at 1.1 |
| `minNeighbors` | Higher = fewer false positives, fewer detections | 4–15 | Start at 10 |
| `minSize` | Ignore small detections | Based on your target | Typical: 40–80 px |
| `maxSize` | Ignore large detections | Based on your target | Typical: 150–300 px |

### Training Parameters (in `cascadeutils.py`)

Edit these to retrain the cascade:

```python
# opencv_traincascade command parameters
numPos = 500           # Number of positive samples
numNeg = 1000          # Number of negative samples
numStages = 15         # Number of cascade stages
minHitRate = 0.995     # Minimum detection rate per stage
maxFalseAlarmRate = 0.5  # Maximum false alarm rate
```

---

## Development

### Code Modules

#### `main.py` — Bot Main Loop

- Captures screenshots from the OSRS window
- Runs cascade detection on each frame
- Calculates click points from detections
- Executes bot actions in a separate thread
- Displays visualization (optional)

**Key Functions**:
- `bot_actions(rectangles)`: Clicks on detected targets
- `detectMultiScale()`: OpenCV cascade detection
- `get_click_points()`: Converts rectangles to clickable points

#### `vision.py` — Vision Utilities

Utility class for vision operations:

- `get_click_points(rectangles)`: Converts [x,y,w,h] rectangles to [x,y] click points (center of each rectangle)
- `draw_rectangles(img, rectangles)`: Draws green bounding boxes
- `draw_crosshairs(img, points)`: Draws magenta crosshair markers on click points
- `centroid(point_list)`: Calculates geometric center of multiple points

**Usage**:
```python
from vision import Vision
vision = Vision()

# Get click coordinates from detected rectangles
click_points = vision.get_click_points(rectangles)

# Draw visualizations for debugging
img = vision.draw_rectangles(img, rectangles)
img = vision.draw_crosshairs(img, click_points)
```

#### `windowcapture.py` — Window Capture

Captures screenshots from a named window with full multi-monitor support:

- Finds the OSRS window by name
- Accounts for window borders and title bars
- Converts image coordinates back to screen coordinates
- **Handles negative coordinates on secondary monitors**
- **Refreshes position if window moves between monitors**

**Key Methods**:
- `__init__(window_name)`: Finds and initializes window handle
- `get_screenshot()`: Returns NumPy array of window contents
- `get_screen_position(img_point)`: Converts image coordinates to screen coordinates (handles negative coords)
- `refresh_window_position()`: Updates window position (useful if window moves between monitors)
- `list_window_names()`: Static method to list all visible windows and their handles

**Multi-Monitor Notes**:
- Screen coordinates can be **negative** if monitor is positioned to the left of the primary monitor
- The bot automatically refreshes window position every 60 frames to handle window movement
- You can manually call `refresh_window_position()` for more frequent updates if needed

**Usage**:
```python
from windowcapture import WindowCapture

wincap = WindowCapture('Old School RuneScape')
screenshot = wincap.get_screenshot()  # NumPy array (H, W, 3)
screen_pos = wincap.get_screen_position((100, 100))  # (screen_x, screen_y) - may be negative!

# If window moves between monitors, refresh position
wincap.refresh_window_position()

# List available windows to find exact window names
WindowCapture.list_window_names()
```

#### `cascadeutils.py` — Training Utilities

Utilities for cascade classifier training:

- Generates `pos.vec` from annotated positive images
- Trains cascade classifiers using `opencv_traincascade`
- Tests detection quality on held-out images
- Compares before/after training metrics

**Command-line Usage**:
```bash
python cascadeutils.py --generate-vec
python cascadeutils.py --train
python cascadeutils.py --test-detection
```

#### `edgefilter.py` — Image Preprocessing

Image preprocessing utilities:

- Edge detection (Canny, Sobel)
- Contrast enhancement
- Histogram equalization
- Blur and noise reduction

**Usage**:
```python
from edgefilter import EdgeFilter
# Details in edgefilter.py source
```

### Performance Metrics (KPIs)

See [`docs/KPIs.md`](docs/KPIs.md) for detailed performance metrics and improvement goals:

1. **Detection Quality**: Precision/recall on annotated test sets
2. **Click Accuracy**: Percentage of clicks landing on valid targets
3. **Throughput & Latency**: FPS and detection-to-click time
4. **Bot Reliability**: Session length until crash/manual exit
5. **Training Data Health**: Balance, variety, and annotation quality
6. **Tests & Safety**: Unit tests for click math, window offsets, etc.
7. **Setup & Reproducibility**: Clean setup and repeatable retraining

**Measuring Improvements**:

```bash
# Before making changes, establish baseline metrics
# Run the bot for a fixed time and record:
# - Detection rate (targets found per minute)
# - Click accuracy (correct clicks / total clicks)
# - False positive rate (clicks on non-targets)
# - FPS (frames per second)

# After improvements:
# - Compare metrics vs. baseline
# - Document what changed and why it helped
```

---

## Troubleshooting

### Bot Won't Start

**Error**: `Window not found: Old School RuneScape`

**Solution**:
1. Ensure OSRS is running
2. Check the exact window title (may vary by OSRS client version)
3. Update `WindowCapture('Old School RuneScape')` with the correct title:

```python
# Find the actual window title:
import win32gui
windows = [win32gui.GetWindowText(h) for h in win32gui.ListWindows()]
print(windows)  # Find OSRS in the list

# Update main.py with the exact title
wincap = WindowCapture('Actual Window Title')
```

### No Detections Found

**Problem**: Bot runs but doesn't detect targets

**Diagnosis & Solutions**:

1. **Verify cascade file exists**:
   ```bash
   ls cascade/cascade.xml
   ```
   If missing, retrain using `cascadeutils.py`.

2. **Check detection parameters**:
   ```python
   # In main.py, make detection more lenient:
   rectangles = cascade_cow.detectMultiScale(
       gray_image,
       scaleFactor=1.05,   # Lower = more thorough (slower)
       minNeighbors=5,     # Lower = more detections (more false positives)
       minSize=(30, 30),   # Lower = catch smaller objects
       maxSize=(200, 200)  # Higher = catch larger objects
   )
   ```

3. **Verify training data quality**:
   - Re-examine positive and negative samples
   - Ensure annotations are accurate (no missed objects)
   - Check that lighting/angles match your bot's runtime environment

4. **Retrain cascade with better data**:
   ```bash
   python cascadeutils.py --generate-vec
   python cascadeutils.py --train
   ```

### Detections Are Inaccurate (False Positives)

**Problem**: Bot clicks on wrong objects or background

**Solutions**:

1. **Increase detection stringency**:
   ```python
   rectangles = cascade_cow.detectMultiScale(
       gray_image,
       scaleFactor=1.15,   # Higher = fewer detections
       minNeighbors=12,    # Higher = more confident detections
   )
   ```

2. **Improve negative training data**:
   - Add more diverse non-target images (backgrounds, UI, other objects)
   - Ensure negative samples match the bot's runtime environment

3. **Retrain with better parameters**:
   ```python
   # In cascadeutils.py:
   minHitRate = 0.99      # Stricter detection threshold
   maxFalseAlarmRate = 0.1  # Lower false positive tolerance
   ```

### Bot Clicks in Wrong Location (Multi-Monitor Issue)

**Problem**: Bot coordinates are off, especially on multi-monitor setups

**Diagnosis**:

1. **Verify window is on correct monitor**:
   ```python
   from windowcapture import WindowCapture
   wincap = WindowCapture('Old School RuneScape')
   print(f"Window position: offset_x={wincap.offset_x}, offset_y={wincap.offset_y}")
   # On secondary monitors, offset values may be negative
   ```

2. **Check window title is exact**:
   ```python
   # List all windows to find exact name
   WindowCapture.list_window_names()
   ```

3. **Verify visualization is correct**:
   - Run bot in detection-only mode (comment out click commands)
   - Check if drawn rectangles and crosshairs align with actual targets
   - If visualization is wrong, the issue is with coordinate conversion

**Solutions**:

1. **Manually refresh window position**:
   ```python
   # In main.py, add more frequent refreshes:
   REFRESH_WINDOW_INTERVAL = 30  # Instead of 60
   ```

2. **Move OSRS window to primary monitor**:
   - Multi-monitor setups with negative coordinates work, but primary monitor is more reliable
   - Avoid windows that span multiple monitors

3. **Test with known coordinates**:
   ```python
   # Add debug output in main.py to verify coordinates
   targets = vision_cow.get_click_points(rectangles)
   if targets:
       screen_pos = wincap.get_screen_position(targets[0])
       print(f"Image: {targets[0]}, Screen: {screen_pos}")
   ```

### Performance Issues (Low FPS, Lag)

**Problem**: Bot is slow or detection is taking too long

**Solutions**:

1. **Reduce image resolution**:
   ```python
   # Downscale before detection
   scale = 0.5
   small = cv.resize(screenshot, (0, 0), fx=scale, fy=scale)
   rectangles = cascade.detectMultiScale(small)
   # Scale coordinates back up
   rectangles = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) 
                 for (x, y, w, h) in rectangles]
   ```

2. **Increase scaleFactor** (less thorough but faster):
   ```python
   scaleFactor=1.2  # Default 1.1; higher = faster
   ```

3. **Profile the code**:
   ```python
   import time
   t0 = time.time()
   # ... bot code ...
   print(f"Detection: {time.time() - t0:.3f}s")
   ```

### Matches Window Freezes or Goes Black (Fullscreen Mode)

**Problem**: The display window freezes or becomes unresponsive when OSRS switches to fullscreen

**Background**: When OSRS enters fullscreen mode, Windows prevents other applications from capturing or displaying over it. The bot now handles this gracefully and automatically recovers.

**What's happening**: 
- When OSRS goes fullscreen, capture fails temporarily
- The display window shows the last successful frame (or a placeholder) to avoid going black
- Console prints "Capture failed #N" to track failures
- After 5 consecutive failures, the bot automatically reinitializes window capture
- Once OSRS returns to windowed mode, capture resumes automatically
- You'll see a red "CAPTURE FAILING (N)" overlay indicating the number of consecutive failures

**Expected behavior (seamless transition)**:
1. You're in windowed mode, bot is running normally
2. You switch OSRS to fullscreen
3. Display shows last frame + red failure counter overlay
4. After ~5 frames, bot reinitializes and keeps trying
5. You switch OSRS back to windowed mode
6. Capture resumes immediately, display returns to normal
7. No manual intervention needed

**Solutions**:

1. **For best experience: keep OSRS in windowed/borderless mode**:
   - In OSRS Settings → Display, select "Windowed" or "Resizable Window" instead of "Fullscreen"
   - Provides uninterrupted bot operation and better visibility
   - Recommended for any automation setup

2. **If you must use fullscreen**:
   - The bot will continue running and automatically handle the transition
   - Hotkey saves (f/d) continue to work even during capture unavailability
   - Once you return to windowed mode, everything resumes normally
   - No data is lost; frames are cached and reused during fullscreen periods

3. **Check for persistent issues**:
   - Look at the "CAPTURE FAILING" counter in the display
   - If it keeps increasing beyond 5, there may be a deeper issue:
     ```python
     # Increase max failures in main.py:
     MAX_CAPTURE_FAILURES_BEFORE_REINIT = 10  # Try up to 10 times
     ```
   - Check console for detailed error messages

### Cascade Training Fails

**Error**: `opencv_traincascade` not found or training crashes

**Solutions**:

1. **Verify OpenCV installation**:
   ```bash
   python -c "import cv2; print(cv2.__version__)"
   ```

2. **Check training data paths**:
   - Ensure `pos.txt` and `neg.txt` reference existing files
   - Use absolute or working-directory-relative paths consistently

3. **Increase resources**:
   - Cascade training is memory-intensive
   - Close other applications
   - Use smaller dataset if machine has <8GB RAM

4. **Update training parameters** in `cascadeutils.py`:
   ```python
   numPos = 100   # Start small, then increase
   numNeg = 200
   numStages = 10  # Fewer stages = faster training
   ```

---

## Safety & Ethics

### Important Notes

⚠️ **Game Terms of Service**: Using automation bots may violate OSRS's Terms of Service. Use this project **for learning purposes only** and assume all responsibility for violations.

### Safe Practices

1. **Test in offline/private environments** first
2. **Use detection-only mode** (visualize without clicking) to verify accuracy
3. **Limit bot runtime** to avoid detection by anti-cheat systems
4. **Monitor bot behavior** — never leave unattended
5. **Respect game communities** — automation can impact game experience
6. **Document your training data** — ensure you own rights to any images used

### Debugging Without Game Interaction

To test detection without clicking on the game:

```python
# In main.py, comment out bot_actions:
# bot_actions(rectangles)  # DISABLED FOR TESTING

# Instead, visualize detections:
screenshot = vision_cow.draw_rectangles(screenshot, rectangles)
screenshot = vision_cow.draw_crosshairs(screenshot, 
                                        vision_cow.get_click_points(rectangles))
cv.imshow('Detections', screenshot)
```

---

## Quick Reference

### Common Commands

```bash
# Setup
conda env create -f environment.yml
conda activate opencv-runescape-env

# Run bot
cd cascade_classifier
python main.py

# Retrain cascade
python cascadeutils.py --generate-vec
python cascadeutils.py --train

# Test detection
python cascadeutils.py --test-detection
```

### File Checklist Before Running

- [ ] `cascade/cascade.xml` exists
- [ ] OSRS is running
- [ ] Window title matches `WindowCapture()` call
- [ ] Python environment is activated
- [ ] Working directory is `cascade_classifier/`

### Performance Targets (from KPIs.md)

- **Detection**: 0.9+ precision and recall on test set
- **Click accuracy**: 85%+ correct clicks
- **FPS**: 10–30 FPS on typical hardware
- **Session length**: 30+ minutes without crashes
- **Data balance**: 3:1 negative-to-positive ratio

---

## Further Reading

- **OpenCV Cascade Training**: [OpenCV Docs](https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html)
- **Project Metrics**: See [`docs/KPIs.md`](docs/KPIs.md)
- **OSRS Game**: [Old School RuneScape](https://oldschool.runescape.com)

---

## Annotation Tool (opencv_annotation.exe)

This repository uses OpenCV's annotation GUI (`opencv_annotation.exe`) to create positive object annotations for `pos.txt` and `pos.vec` generation. Below are quick steps, common commands, and troubleshooting tips.

Usage (GUI):

1. Open `opencv_annotation.exe` (bundled with OpenCV or available via contrib tools).
2. Set the `--images` folder to `cascade_classifier/positive`.
3. Set the output `--annotations` file to `cascade_classifier/pos.txt`.
4. Draw rectangles around each target instance and save.

Example (command-line):

```powershell
# From the repository root
# Run the annotation GUI and point it at the positive images folder
opencv_annotation.exe --annotations=cascade_classifier/pos.txt --images=cascade_classifier/positive
```

Notes on `pos.txt` format (for `opencv_createsamples`):

Each annotated line should follow the OpenCV format when used for sample creation:

```
positive/cow_001.jpg 1 x y width height
```

Where `x y width height` are the bounding box values for the single object on that line. If an image has multiple boxes, annotate them as one line with the count followed by the boxes.

Generating `pos.vec` (after annotations):

```powershell
cd cascade_classifier
opencv_createsamples -info pos.txt -vec pos.vec -w 24 -h 24
```

Then train with `opencv_traincascade` as described earlier.

Common `opencv_annotation.exe` problems & fixes

- Exit code 1 / GUI fails to start:
   - Reason: Missing Visual C++ redistributable or DLL dependencies. Install the appropriate Microsoft Visual C++ Redistributable matching your OpenCV build (often 2015-2019).
   - Reason: Running from a different working directory; relative paths in the GUI may be resolved incorrectly. Run `opencv_annotation.exe` from the repository root or pass absolute paths to `--images` and `--annotations`.
   - Reason: `opencv_annotation.exe` not in PATH. Provide the full path to the executable or run it from the OpenCV bin folder.

- No images shown in the GUI:
   - Ensure `cascade_classifier/positive` contains image files (png/jpg) and that the paths are readable.
   - Check for spaces or non-ASCII characters in paths—use short paths or move images to a simple path.

- Annotations saved but `opencv_createsamples` fails:
   - Verify `pos.txt` lines match the expected format and that files exist.
   - Use absolute paths in `pos.txt` if relative paths fail.

If `opencv_annotation.exe` returns an error code and no helpful message, run it from PowerShell or Command Prompt to capture stderr output:

```powershell
.\path\to\opencv_annotation.exe --annotations=C:\full\path\to\cascade_classifier\pos.txt --images=C:\full\path\to\cascade_classifier\positive
```

Alternative annotation tools

- LabelImg: Friendly GUI that exports Pascal VOC XML; convert or write a small script to convert VOC to the `pos.txt` format.
- MakeSense.ai (web-based): Export annotations and convert as needed.

---

## License & Attribution

This project is for **educational purposes**. Use responsibly and in compliance with OSRS Terms of Service.

---

**Last Updated**: January 2026  
**Course Context**: Image Processing Fundamentals (USF, Professor Dimitry Goldgof)
