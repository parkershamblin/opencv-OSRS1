# opencv-OSRS1

`opencv-OSRS1` is a small Windows-only OpenCV experiment that detects cows in Old School RuneScape and clicks them.

Under the hood it uses:
- a trained cascade classifier in `cascade_classifier/cascade/cascade.xml`
- Win32 window capture plus `Pillow.ImageGrab` to grab the game client
- `pyautogui` to move the mouse and click the first detection

This is an experiment, not a polished bot. It has hard-coded assumptions, no automated tests, and it will take over your mouse while running.

Automation may violate game rules. Use it at your own risk.

## Demo
![opencv-OSRS-demo](docs/img/opencv-OSRS-demo.gif)

## Repository Layout
- `cascade_classifier/main.py`: main detection loop and click behavior
- `cascade_classifier/windowcapture.py`: window lookup and screenshot capture
- `cascade_classifier/vision.py`: rectangle drawing and click-point helpers
- `cascade_classifier/cascadeutils.py`: helper for rebuilding `neg.txt`
- `cascade_classifier/cascade/`: trained cascade output
- `docs/training-image-manifest.md`: original training image inventory
- `docs/KPIs.md`: notes from the 2026 rebuild

## Quick Start

### 1) Create a Python environment
Use Windows CPython 3.12 if you want the least annoying setup.

Avoid MSYS2/UCRT Python builds such as `C:\msys64\ucrt64\bin\python.exe`; some packages used here are not published for that interpreter.

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If your Python install creates `.venv/bin` instead of `.venv/Scripts`, use `.\.venv\bin\Activate.ps1`.

Conda is also supported:

```powershell
conda env create -f environment.yml
conda activate opencv-automation
```

### 2) Run the bot
Before you start:
- open the game client
- make sure the window title is exactly `Old School Runescape`
- expect the script to move the mouse and click on its own

From the repository root:

```powershell
python cascade_classifier\main.py
```

Runtime controls:
- `q`: quit
- `f`: save the current screenshot to `cascade_classifier/positive/`
- `d`: save the current screenshot to `cascade_classifier/negative/`

## Retraining Workflow
You do not need these steps to run the existing model. You only need them if you want to rebuild the dataset or retrain the cascade.

### 1) Restore the original training image set
The initial screenshot set can be restored from commit `7257565`.

```bash
git checkout 7257565 -- cascade_classifier/positive cascade_classifier/negative
```

Exact file list:
- [docs/training-image-manifest.md](docs/training-image-manifest.md)

### 2) Rebuild `neg.txt`
Run this from `cascade_classifier`:

```powershell
Set-Location cascade_classifier
python -c "from cascadeutils import generate_negative_description_file; generate_negative_description_file()"
```

### 3) Annotate positive images into `pos.txt`
You need the OpenCV command-line tools for this step:
- `opencv_annotation.exe`
- `opencv_createsamples.exe`
- `opencv_traincascade.exe`

Important: these tools do not come from `pip install opencv-python`. You need an OpenCV build that includes the training utilities.

Example:

```bash
opencv_annotation.exe --annotations=pos.txt --images=positive/
```

Annotation controls:
- click once for the top-left corner and once for the bottom-right corner
- press `c` to confirm a box
- press `d` to undo the last box
- press `n` for the next image
- press `esc` to exit

### 4) Create vector samples
```bash
opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec
```

### 5) Train the cascade
Base command:

```bash
opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -numPos 200 -numNeg 100 -numStages 10 -w 24 -h 24
```

Final command recorded for this project:

```bash
opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -precalcValBufSize 6000 -precalcIdxBufSize 6000 -numPos 200 -numNeg 1000 -numStages 12 -w 24 -h 24 -maxFalseAlarmRate 0.4 -minHitRate 0.999
```

The trained model is written to `cascade_classifier/cascade/cascade.xml`.

## Current Limitations
- Windows only: capture code depends on Win32 APIs.
- `main.py` uses a hard-coded window title.
- The bot clicks the first detected rectangle and then sleeps for 10 seconds.
- There is no config file, CLI, or automated test suite yet.
- Retraining is manual and depends on external OpenCV binaries.

## Project Notes
- January 2026 rebuild notes and KPIs: [docs/KPIs.md](docs/KPIs.md)

## Credits
Conceptual inspiration came from Learn Code By Gaming's [OpenCV Object Detection in Games playlist](https://www.youtube.com/watch?v=KecMlLUuiE4&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI).
