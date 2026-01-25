## January 2026 Update
I’m rebuilding this Old School RuneScape computer vision bot while taking “Image Processing Fundamentals” with Professor Dimitry Goldgof at USF. The class is giving me ideas, and I’m trying them here.

### Course snapshot
![Course overview](/img/image_processing_fundaementals_overview.jpg)

###  Demo
![opencv-OSRS-demo](img/opencv-OSRS-demo.gif)

### (Pending) Updated Demo
> Pending clip of updated project demo.

## KPI Dashboard
These are the checkpoints I’m watching so I can tell if the bot is actually getting better.

### 1) Detection quality
- **Goal:** better precision/recall on a held-out batch of annotated screenshots.
- **Plan:** clean up `positive/` and `negative/`, regenerate `pos.vec`, retrain the cascade.
- **Measure:** compare detections vs. labels before/after retraining.

### 2) Click accuracy
- **Goal:** more clicks land on cows, not dirt or UI.
- **Plan:** sample `vision.get_click_points()` during runs, review screenshots.
- **Measure:** correct clicks / total clicks per session; adjust box-to-click math if it drifts.

### 3) Throughput and latency
- **Goal:** keep the loop feeling real-time.
- **Plan:** profile around `detectMultiScale` through the PyAutoGUI click.
- **Measure:** FPS and detection-to-click latency each run.

### 4) Bot reliability
- **Goal:** longer runs without crashes or stalls.
- **Plan:** treat dialog stalls and misclick loops as real bugs; watch `bot_actions` threads.
- **Measure:** session length until `cv.waitKey` exit or crash, plus notes on what killed it.

### 5) Training data health
- **Goal:** a varied, well-labeled set across lighting, zoom, and angles.
- **Plan:** audit `positive/` and `negative/` for dupes/imbalance; refresh `pos.txt` before retrains.
- **Measure:** quick balance/annotation checks each refresh cycle.

### 6) Tests and safety net
- **Goal:** keep click math and capture offsets from regressing.
- **Plan:** unit tests for `get_click_points`, `get_screen_position`, rectangle drawing, plus a smoke test that loads `cascade.xml`.
- **Measure:** green test run before I touch the bot.

### 7) Setup and reproducibility
- **Goal:** easy setup on Windows and retrains that still work later.
- **Plan:** keep `requirements.txt` current; keep the OpenCV training commands together in `cascadeutils.py` or the README.
- **Measure:** clean-machine setup check and a retrain walkthrough that actually runs.

---

## Why this matters
These keep me honest: better detections, accurate clicks, real-time loops, stable sessions, healthy data, safety nets, and setup that doesn’t break. It’s how I turn class theory into a project I’d actually show people.

[1]: https://github.com/parkershamblin/opencv-OSRS1 "GitHub - parkershamblin/opencv-OSRS1: Created a Bot to farm cows in Old School RuneScape using OpenCV (Project created in 2020 but reuploaded in 2022)."
[2]: https://www.geeksforgeeks.org/opencv-projects-ideas-for-beginners/?utm_source=chatgpt.com "15 OpenCV Projects Ideas for Beginners to Practice in 2025 - GeeksforGeeks"
