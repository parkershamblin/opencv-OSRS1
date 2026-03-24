# opencv-OSRS
This program automates cow farming in Old School RuneScape by leveraging OpenCV and a custom-trained cascade classifier model, which I trained on images of RuneScape gameplay. PyAutoGUI is used to interact with the game through mouse actions, while the image processing runs in real time.

## Demo:
![opencv-OSRS-demo](docs/img/opencv-OSRS-demo.gif)

## Reproducible Setup

### 1) Python environment
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2) Restore the original training image set
The initial training set used for the early model can be restored from commit `7257565`.

```bash
git checkout 7257565 -- cascade_classifier/positive cascade_classifier/negative
```

Dataset manifest and exact file list:
- [docs/training-image-manifest.md](docs/training-image-manifest.md)

### 3) Generate the negative description file (`neg.txt`)
Run from the `cascade_classifier` folder:

```bash
cd cascade_classifier
python -c "from cascadeutils import generate_negative_description_file; generate_negative_description_file()"
```

### 4) Manually annotate positives (`pos.txt`)
Install/build OpenCV tools so these executables are available:
- `opencv_annotation.exe`
- `opencv_createsamples.exe`
- `opencv_traincascade.exe`

Example command:

```bash
opencv_annotation.exe --annotations=pos.txt --images=positive/
```

Annotation controls:
- Click once for top-left corner and once for bottom-right corner.
- Press `c` to confirm a box.
- Press `d` to undo last box.
- Press `n` for next image.
- Press `esc` to exit.

### 5) Create vector samples (`pos.vec`)
```bash
opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec
```

### 6) Train the cascade model
Base command:

```bash
opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -numPos 200 -numNeg 100 -numStages 10 -w 24 -h 24
```

Final command used in this project:

```bash
opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -precalcValBufSize 6000 -precalcIdxBufSize 6000 -numPos 200 -numNeg 1000 -numStages 12 -w 24 -h 24 -maxFalseAlarmRate 0.4 -minHitRate 0.999
```

Trained model output is written to:
- `cascade_classifier/cascade/cascade.xml`

### 7) Run inference bot
From `cascade_classifier`:

```bash
python main.py
```

Notes:
- `main.py` expects the game window title to be exactly `Old School Runescape`.
- While running, press `q` in the OpenCV window to quit.
- During collection mode, press `f` to save positive screenshots and `d` to save negative screenshots.

## Updates
- Jan 2026: ["Imaging Processing Fundamentals" & KPIs](docs/KPIs.md)

## Credits:
Conceptual inspiration drawn from: [Learn Code By Gaming's "OpenCV Object Detection in Games" Playlist](https://www.youtube.com/watch?v=KecMlLUuiE4&list=PL1m2M8LQlzfKtkKq2lK5xko4X-8EZzFPI).

