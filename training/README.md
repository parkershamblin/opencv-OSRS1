# Training Datasets

Each dataset folder keeps one camera/zoom setup separate from the others.

- `fixed_zoom_v1/positive`: fixed-camera adult cow screenshots.
- `fixed_zoom_v1/negative`: fixed-camera screenshots with no adult cows.
- `wildcard_zoom/`: older mixed-camera data kept separate for reference.

Use the fixed-zoom dataset for the next cascade pass:

```powershell
cd cascade_classifier
python main.py
```

Choose collect mode, then select `fixed_zoom_v1`.

After collecting and annotating:

```powershell
python cascadeutils.py --dataset fixed_zoom_v1 --annotate
python cascadeutils.py --dataset fixed_zoom_v1 --generate-neg
python cascadeutils.py --dataset fixed_zoom_v1 --generate-hard-negatives
python cascadeutils.py --dataset fixed_zoom_v1 --generate-vec --num 225 --window-width 80 --window-height 64
python cascadeutils.py --dataset fixed_zoom_v1 --train --cascade-dir cascade_adult_cow_fixed_zoom_v1 --num-pos 200 --num-neg 1600 --num-stages 14 --feature-type LBP --max-false-alarm-rate 0.25 --window-width 80 --window-height 64
```
