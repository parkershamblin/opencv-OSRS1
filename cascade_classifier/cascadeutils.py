import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
TRAINING_DIR = REPO_DIR / 'training'
OPENCV_SOURCE_DIR = REPO_DIR / 'opencv'
OPENCV_APPS_BUILD_DIR = REPO_DIR / 'opencv_build' / 'cascade_apps'
DATASET_DIR = BASE_DIR
POSITIVE_DIR = DATASET_DIR / 'positive'
NEGATIVE_DIR = DATASET_DIR / 'negative'
POS_FILE = DATASET_DIR / 'pos.txt'
NEG_FILE = DATASET_DIR / 'neg.txt'
VEC_FILE = DATASET_DIR / 'pos.vec'
VEC_META_FILE = DATASET_DIR / 'pos.vec.meta'
CASCADE_DIR = BASE_DIR / 'cascade_adult_cow'
WINDOW_SIZE = (80, 64)
MIN_POSITIVE_BOXES = 100
RECOMMENDED_POSITIVE_BOXES = 150
DEFAULT_NUM_STAGES = 14
DEFAULT_FEATURE_TYPE = 'LBP'
DEFAULT_MAX_FALSE_ALARM_RATE = 0.25
ANNOTATION_MAX_WINDOW_HEIGHT = 650
ANNOTATION_RESIZE_FACTOR = 2
ANNOTATION_WINDOW_POLL_SECONDS = 0.25
ANNOTATION_WINDOW_RECT_TOLERANCE = 8
IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png'}
HARD_NEGATIVE_PREFIX = 'hardneg_'
UI_NEGATIVE_REGIONS = [
    ('bottom_all', 0.00, 0.84, 1.00, 1.00),
    ('chat', 0.00, 0.84, 0.48, 1.00),
    ('actions', 0.58, 0.84, 1.00, 1.00),
    ('minimap', 0.78, 0.00, 1.00, 0.34),
    ('orbs', 0.78, 0.00, 0.94, 0.38),
]


def _exe_name(name):
    return f'{name}.exe' if os.name == 'nt' else name


def resolve_dataset_dir(dataset):
    path = Path(dataset)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == TRAINING_DIR.name:
        return (REPO_DIR / path).resolve()
    return (TRAINING_DIR / path).resolve()


def set_training_dataset(dataset):
    if not dataset:
        return

    global DATASET_DIR, POSITIVE_DIR, NEGATIVE_DIR, POS_FILE, NEG_FILE, VEC_FILE, VEC_META_FILE
    DATASET_DIR = resolve_dataset_dir(dataset)
    POSITIVE_DIR = DATASET_DIR / 'positive'
    NEGATIVE_DIR = DATASET_DIR / 'negative'
    POS_FILE = DATASET_DIR / 'pos.txt'
    NEG_FILE = DATASET_DIR / 'neg.txt'
    VEC_FILE = DATASET_DIR / 'pos.vec'
    VEC_META_FILE = DATASET_DIR / 'pos.vec.meta'
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)


def find_opencv_executable(name):
    exe_name = _exe_name(name)
    candidates = [
        shutil.which(exe_name),
        OPENCV_APPS_BUILD_DIR / 'bin' / 'Release' / exe_name,
        OPENCV_APPS_BUILD_DIR / 'bin' / exe_name,
        OPENCV_APPS_BUILD_DIR / 'apps' / 'traincascade' / 'Release' / exe_name,
        OPENCV_APPS_BUILD_DIR / 'apps' / 'createsamples' / 'Release' / exe_name,
        OPENCV_APPS_BUILD_DIR / 'apps' / 'annotation' / 'Release' / exe_name,
        REPO_DIR / 'opencv_build' / 'bin' / 'Release' / exe_name,
        REPO_DIR / 'opencv_build' / 'bin' / exe_name,
        REPO_DIR / 'opencv_build' / 'apps' / 'traincascade' / 'Release' / exe_name,
        REPO_DIR / 'opencv_build' / 'apps' / 'createsamples' / 'Release' / exe_name,
        REPO_DIR / 'opencv_build' / 'apps' / 'annotation' / 'Release' / exe_name,
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)

    return None


def generate_negative_description_file():
    with NEG_FILE.open('w') as f:
        for path in sorted(NEGATIVE_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                f.write(f'negative/{path.name}\n')


def negative_source_images():
    return [
        path
        for path in sorted(NEGATIVE_DIR.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.startswith(HARD_NEGATIVE_PREFIX)
    ]


def _import_cv2():
    try:
        import cv2 as cv
    except ImportError as exc:
        raise RuntimeError(
            'OpenCV Python is required for --generate-hard-negatives. '
            'Run this from the opencv-runescape-env environment or install opencv-python.'
        ) from exc
    return cv


def scaled_crop_bounds(image_width, image_height, region):
    _, x1_ratio, y1_ratio, x2_ratio, y2_ratio = region
    x1 = max(0, min(image_width, round(image_width * x1_ratio)))
    y1 = max(0, min(image_height, round(image_height * y1_ratio)))
    x2 = max(0, min(image_width, round(image_width * x2_ratio)))
    y2 = max(0, min(image_height, round(image_height * y2_ratio)))
    return x1, y1, x2, y2


def deterministic_gameplay_crop_bounds(image_width, image_height, crop_count, window_size):
    gameplay_width = round(image_width * 0.78)
    gameplay_height = round(image_height * 0.84)
    crop_widths = [
        max(window_size[0] * 2, round(image_width * 0.16)),
        max(window_size[0] * 3, round(image_width * 0.22)),
    ]
    crop_heights = [
        max(window_size[1] * 2, round(image_height * 0.16)),
        max(window_size[1] * 3, round(image_height * 0.22)),
    ]
    anchors = [
        (0.05, 0.05),
        (0.25, 0.05),
        (0.45, 0.05),
        (0.62, 0.10),
        (0.05, 0.28),
        (0.28, 0.30),
        (0.50, 0.32),
        (0.08, 0.55),
        (0.34, 0.58),
        (0.58, 0.58),
    ]

    bounds = []
    for index, (x_ratio, y_ratio) in enumerate(anchors[:crop_count]):
        crop_width = min(gameplay_width, crop_widths[index % len(crop_widths)])
        crop_height = min(gameplay_height, crop_heights[index % len(crop_heights)])
        max_x = max(0, gameplay_width - crop_width)
        max_y = max(0, gameplay_height - crop_height)
        x1 = round(max_x * x_ratio)
        y1 = round(max_y * y_ratio)
        bounds.append((x1, y1, x1 + crop_width, y1 + crop_height))

    return bounds


def write_crop(cv, image, output_path, bounds, min_size, overwrite=False):
    x1, y1, x2, y2 = bounds
    if x2 - x1 < min_size[0] or y2 - y1 < min_size[1]:
        return False
    if output_path.exists() and not overwrite:
        return False

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    return bool(cv.imwrite(str(output_path), crop))


def generate_hard_negative_crops(
    scenery_crops_per_image=6,
    max_source_images=None,
    overwrite=False,
    window_size=WINDOW_SIZE,
):
    cv = _import_cv2()
    source_images = negative_source_images()
    if max_source_images:
        source_images = source_images[:max_source_images]

    if not source_images:
        raise RuntimeError(f'No source negative images found in {NEGATIVE_DIR}.')

    min_size = window_size
    written = 0
    skipped = 0
    failed = 0

    for source_path in source_images:
        image = cv.imread(str(source_path))
        if image is None:
            failed += 1
            continue

        image_height, image_width = image.shape[:2]
        crop_jobs = []
        for region in UI_NEGATIVE_REGIONS:
            label = region[0]
            crop_jobs.append((label, scaled_crop_bounds(image_width, image_height, region)))

        for index, bounds in enumerate(
            deterministic_gameplay_crop_bounds(
                image_width,
                image_height,
                scenery_crops_per_image,
                window_size,
            ),
            1,
        ):
            crop_jobs.append((f'scenery_{index:02d}', bounds))

        for label, bounds in crop_jobs:
            output_name = f'{HARD_NEGATIVE_PREFIX}{source_path.stem}_{label}.jpg'
            output_path = NEGATIVE_DIR / output_name
            if output_path.exists() and not overwrite:
                skipped += 1
                continue
            if write_crop(cv, image, output_path, bounds, min_size, overwrite):
                written += 1
            else:
                skipped += 1

    generate_negative_description_file()
    neg_count = len([line for line in NEG_FILE.read_text().splitlines() if line.strip()])
    print(
        f'Generated {written} hard-negative crops '
        f'({skipped} skipped, {failed} unreadable source images).'
    )
    print(f'Regenerated neg.txt with {neg_count} negative images.')


def normalize_positive_description_file():
    if not POS_FILE.exists():
        return

    normalized = []
    changed = False
    for line in POS_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            normalized.append(line)
            continue

        path = Path(parts[0])
        if path.is_absolute():
            try:
                path = path.relative_to(DATASET_DIR)
            except ValueError:
                try:
                    path = path.relative_to(POSITIVE_DIR.parent)
                except ValueError:
                    path = POSITIVE_DIR / path.name
            parts[0] = path.as_posix()
            changed = True
        else:
            parts[0] = path.as_posix()

        normalized.append(' '.join(parts))

    if changed:
        POS_FILE.write_text('\n'.join(normalized) + '\n')
        print('Normalized pos.txt image paths to be relative to cascade_classifier/.')


def positive_box_count():
    if not POS_FILE.exists():
        return 0

    count = 0
    for line in POS_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            count += int(parts[1])
        except ValueError:
            continue
    return count


def resolve_cascade_dir(cascade_dir):
    path = Path(cascade_dir)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def ensure_safe_generated_dir(path):
    base = BASE_DIR.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f'Refusing to clean output outside cascade_classifier/: {resolved}') from exc

    protected = {
        BASE_DIR.resolve(),
        POSITIVE_DIR.resolve(),
        NEGATIVE_DIR.resolve(),
    }
    if resolved in protected:
        raise RuntimeError(f'Refusing to clean protected directory: {resolved}')


def cascade_dir_has_training_output(cascade_dir):
    if not cascade_dir.exists():
        return False

    for path in cascade_dir.iterdir():
        if path.name == 'cascade.xml':
            return True
        if path.name == 'params.xml':
            return True
        if path.name.startswith('stage') and path.suffix == '.xml':
            return True

    return False


def prepare_cascade_output_dir(cascade_dir, clean_output=False):
    if clean_output:
        ensure_safe_generated_dir(cascade_dir)
        if cascade_dir.exists():
            shutil.rmtree(cascade_dir)
        cascade_dir.mkdir(parents=True, exist_ok=True)
        return

    if cascade_dir_has_training_output(cascade_dir):
        raise RuntimeError(
            f'{cascade_dir} already contains cascade training output. '
            'Use --cascade-dir with a fresh folder name, or use --clean-output '
            'when you intentionally want to replace that generated model.'
        )

    cascade_dir.mkdir(parents=True, exist_ok=True)


def vec_metadata():
    if not VEC_META_FILE.exists():
        return None

    metadata = {}
    for line in VEC_META_FILE.read_text().splitlines():
        key, _, value = line.partition('=')
        if key and value:
            metadata[key] = value
    return metadata


def validate_training_files():
    normalize_positive_description_file()

    if not POS_FILE.exists():
        raise RuntimeError('pos.txt is missing. Annotate adult cows before training.')
    if not NEG_FILE.exists():
        raise RuntimeError('neg.txt is missing. Run --generate-neg before training.')

    pos_boxes = 0
    malformed = []
    missing = []
    for line_number, line in enumerate(POS_FILE.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) < 2:
            malformed.append((line_number, 'too few columns'))
            continue
        image_path = Path(parts[0])
        if not image_path.is_absolute():
            image_path = DATASET_DIR / image_path
        if not image_path.exists():
            missing.append((line_number, parts[0]))
        try:
            object_count = int(parts[1])
            if len(parts[2:]) != object_count * 4:
                malformed.append((line_number, 'box count does not match coordinates'))
            pos_boxes += object_count
        except ValueError:
            malformed.append((line_number, 'invalid object count or coordinate'))

    neg_count = len([line for line in NEG_FILE.read_text().splitlines() if line.strip()])

    if missing:
        first_missing = ', '.join(f'line {n}: {p}' for n, p in missing[:5])
        raise RuntimeError(f'pos.txt references missing images: {first_missing}')
    if malformed:
        first_malformed = ', '.join(f'line {n}: {reason}' for n, reason in malformed[:5])
        raise RuntimeError(f'pos.txt has malformed rows: {first_malformed}')
    if pos_boxes < MIN_POSITIVE_BOXES:
        raise RuntimeError(
            f'pos.txt has {pos_boxes} adult-cow boxes; need at least '
            f'{MIN_POSITIVE_BOXES} and preferably {RECOMMENDED_POSITIVE_BOXES}.'
        )
    if neg_count < 100:
        raise RuntimeError(f'neg.txt has only {neg_count} negatives; add more hard negatives first.')

    return pos_boxes, neg_count


def build_opencv_apps():
    existing_apps = [
        find_opencv_executable('opencv_annotation'),
        find_opencv_executable('opencv_createsamples'),
        find_opencv_executable('opencv_traincascade'),
    ]
    if all(existing_apps):
        print('OpenCV cascade apps are already available:')
        for app in existing_apps:
            print(f'  {app}')
        print('Skipping rebuild.')
        return

    if not OPENCV_SOURCE_DIR.exists():
        raise RuntimeError(f'OpenCV source directory not found: {OPENCV_SOURCE_DIR}')

    cmake = shutil.which('cmake')
    if cmake is None:
        raise RuntimeError(
            'cmake was not found on PATH. Install it in this environment with '
            '"conda install -c conda-forge cmake" or add CMake to PATH, then retry.'
        )

    configure_cmd = [
        cmake,
        '-S',
        str(OPENCV_SOURCE_DIR),
        '-B',
        str(OPENCV_APPS_BUILD_DIR),
        '-G',
        'Visual Studio 17 2022',
        '-A',
        'x64',
        '-D',
        'CMAKE_BUILD_TYPE=Release',
        '-D',
        'BUILD_opencv_apps=ON',
        '-D',
        'BUILD_LIST=core,imgproc,imgcodecs,highgui,objdetect,features2d,calib3d,videoio',
        '-D',
        'BUILD_TESTS=OFF',
        '-D',
        'BUILD_PERF_TESTS=OFF',
        '-D',
        'BUILD_EXAMPLES=OFF',
        '-D',
        'BUILD_DOCS=OFF',
        '-D',
        'BUILD_JAVA=OFF',
        '-D',
        'BUILD_opencv_python=OFF',
    ]
    subprocess.run(configure_cmd, cwd=REPO_DIR, check=True)

    build_cmd = [
        cmake,
        '--build',
        str(OPENCV_APPS_BUILD_DIR),
        '--config',
        'Release',
        '--target',
        'opencv_traincascade',
        'opencv_createsamples',
        'opencv_annotation',
    ]
    subprocess.run(build_cmd, cwd=REPO_DIR, check=True)


def run_annotation():
    annotation = find_opencv_executable('opencv_annotation')
    if annotation is None:
        raise RuntimeError('opencv_annotation executable not found. Run --build-apps first.')

    process = subprocess.Popen(
        [
            annotation,
            f'--annotations={POS_FILE}',
            f'--images={POSITIVE_DIR}/',
            f'--maxWindowHeight={ANNOTATION_MAX_WINDOW_HEIGHT}',
            f'--resizeFactor={ANNOTATION_RESIZE_FACTOR}',
        ],
        cwd=DATASET_DIR,
    )
    keep_annotation_window_position_usable(process)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, process.args)


def keep_annotation_window_position_usable(process):
    if os.name != 'nt':
        return

    try:
        import win32api
        import win32con
        import win32gui
        import win32process
    except ImportError:
        print('pywin32 is unavailable; annotation window placement will not be managed.')
        return

    managed_hwnd = None

    while process.poll() is None:
        hwnd = find_process_window(process.pid, win32gui, win32process)
        if hwnd is None:
            time.sleep(ANNOTATION_WINDOW_POLL_SECONDS)
            continue

        current_rect = win32gui.GetWindowRect(hwnd)
        work_rect = monitor_work_rect(hwnd, win32api, win32con)

        if hwnd != managed_hwnd or window_position_needs_correction(current_rect, work_rect):
            left, top = centered_annotation_position(current_rect, work_rect)
            apply_window_position(hwnd, left, top, win32gui, win32con)
            managed_hwnd = hwnd

        time.sleep(ANNOTATION_WINDOW_POLL_SECONDS)


def find_process_window(pid, win32gui, win32process):
    windows = []

    def enum_handler(hwnd, ctx):
        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return
        _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
        if window_pid != pid:
            return
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        if width > 100 and height > 100:
            ctx.append((width * height, hwnd))

    win32gui.EnumWindows(enum_handler, windows)
    if not windows:
        return None

    return max(windows)[1]


def monitor_work_rect(hwnd, win32api, win32con):
    monitor = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
    return win32api.GetMonitorInfo(monitor)['Work']


def centered_annotation_position(rect, work_rect):
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top

    work_left, work_top, work_right, work_bottom = work_rect
    work_width = work_right - work_left
    work_height = work_bottom - work_top

    left = work_left + max(0, (work_width - width) // 2)
    top = work_top + max(0, (work_height - height) // 2)
    return left, top


def window_position_needs_correction(rect, work_rect):
    left, top, right, bottom = rect
    work_left, work_top, work_right, work_bottom = work_rect
    tolerance = ANNOTATION_WINDOW_RECT_TOLERANCE

    return (
        left < work_left - tolerance
        or top < work_top - tolerance
        or right > work_right + tolerance
        or bottom > work_bottom + tolerance
    )


def apply_window_position(hwnd, left, top, win32gui, win32con):
    win32gui.SetWindowPos(
        hwnd,
        None,
        left,
        top,
        0,
        0,
        win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE | win32con.SWP_NOSIZE,
    )


def generate_vec(num_samples=None, window_size=WINDOW_SIZE):
    pos_boxes, _ = validate_training_files()
    createsamples = find_opencv_executable('opencv_createsamples')
    if createsamples is None:
        raise RuntimeError('opencv_createsamples executable not found. Run --build-apps first.')

    sample_count = num_samples or min(pos_boxes, RECOMMENDED_POSITIVE_BOXES)
    if VEC_FILE.exists():
        VEC_FILE.unlink()
    cmd = [
        createsamples,
        '-info',
        str(POS_FILE),
        '-vec',
        str(VEC_FILE),
        '-num',
        str(sample_count),
        '-w',
        str(window_size[0]),
        '-h',
        str(window_size[1]),
    ]
    subprocess.run(cmd, cwd=DATASET_DIR, check=True)
    VEC_META_FILE.write_text(f'num={sample_count}\nw={window_size[0]}\nh={window_size[1]}\n')


def vec_sample_count():
    metadata = vec_metadata()
    if metadata is None:
        return None

    try:
        return int(metadata.get('num', ''))
    except ValueError:
        return None


def validate_vec_window_size(window_size):
    metadata = vec_metadata()
    if metadata is None:
        raise RuntimeError('pos.vec.meta is missing. Regenerate pos.vec with --generate-vec.')

    try:
        vec_width = int(metadata.get('w', ''))
        vec_height = int(metadata.get('h', ''))
    except ValueError as exc:
        raise RuntimeError('pos.vec.meta has invalid width/height. Regenerate pos.vec.') from exc

    if (vec_width, vec_height) != window_size:
        raise RuntimeError(
            f'pos.vec was generated for {vec_width}x{vec_height}, but training requested '
            f'{window_size[0]}x{window_size[1]}. Regenerate with matching '
            '--window-width and --window-height.'
        )

    return None


def train_cascade(
    num_pos=None,
    num_neg=None,
    max_false_alarm_rate=DEFAULT_MAX_FALSE_ALARM_RATE,
    num_stages=DEFAULT_NUM_STAGES,
    feature_type=DEFAULT_FEATURE_TYPE,
    cascade_dir=CASCADE_DIR,
    window_size=WINDOW_SIZE,
    clean_output=False,
):
    pos_boxes, neg_count = validate_training_files()
    traincascade = find_opencv_executable('opencv_traincascade')
    if traincascade is None:
        raise RuntimeError('opencv_traincascade executable not found. Run --build-apps first.')
    if not VEC_FILE.exists():
        raise RuntimeError('pos.vec is missing. Run --generate-vec first.')

    validate_vec_window_size(window_size)
    prepare_cascade_output_dir(cascade_dir, clean_output)
    vec_count = vec_sample_count()
    if vec_count is None:
        raise RuntimeError('pos.vec.meta is missing. Regenerate pos.vec with --generate-vec.')
    train_pos = num_pos or int(vec_count * 0.9)
    if train_pos >= vec_count:
        raise RuntimeError(f'numPos must be less than vec sample count ({vec_count}); got {train_pos}.')
    train_neg = num_neg or min(neg_count, 400)
    cmd = [
        traincascade,
        '-data',
        str(cascade_dir),
        '-vec',
        str(VEC_FILE),
        '-bg',
        str(NEG_FILE),
        '-numPos',
        str(train_pos),
        '-numNeg',
        str(train_neg),
        '-numStages',
        str(num_stages),
        '-w',
        str(window_size[0]),
        '-h',
        str(window_size[1]),
        '-featureType',
        feature_type,
        '-precalcValBufSize',
        '6000',
        '-precalcIdxBufSize',
        '6000',
        '-minHitRate',
        '0.995',
        '-maxFalseAlarmRate',
        str(max_false_alarm_rate),
    ]
    subprocess.run(cmd, cwd=DATASET_DIR, check=True)


def print_commands():
    # Non-static: discover available cascade models in the dataset directory
    print('Adult cow cascade workflow:')
    print('1. python cascadeutils.py --build-apps')
    print('2. python cascadeutils.py --dataset <dataset> --annotate')
    print('3. python cascadeutils.py --dataset <dataset> --generate-neg')
    print('4. python cascadeutils.py --dataset <dataset> --generate-hard-negatives')
    print('5. python cascadeutils.py --dataset <dataset> --generate-vec --num <n> --window-width <w> --window-height <h>')
    print('6. python cascadeutils.py --dataset <dataset> --train --cascade-dir <cascade-dir> --num-pos <n> --clean-output')
    print('')
    print('Available cascade models found:')
    found = []
    # look for cascade directories or xml files in the base dataset dir
    for p in sorted(BASE_DIR.iterdir()):
        if p.is_dir() and p.name.startswith('cascade_'):
            found.append(p.name)
        if p.is_file() and p.suffix.lower() == '.xml' and 'cascade' in p.name.lower():
            found.append(p.name)

    if not found:
        print('  (none found in', str(BASE_DIR) + ')')
    else:
        for name in found:
            print('  ', name)

    print('')
    print('Annotation rule: boxes must cover whole visible adult cow bodies.')
    print('Do not label sheep, baby cows, NPCs, fences, hooves-only, or body fragments as positives.')


def main():
    parser = argparse.ArgumentParser(description='Adult cow cascade training helpers.')
    parser.add_argument('--build-apps', action='store_true')
    parser.add_argument('--annotate', action='store_true')
    parser.add_argument('--generate-neg', action='store_true')
    parser.add_argument('--generate-hard-negatives', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--generate-vec', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--commands', action='store_true')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--num-pos', type=int, default=None)
    parser.add_argument('--num-neg', type=int, default=None)
    parser.add_argument('--num-stages', type=int, default=DEFAULT_NUM_STAGES)
    parser.add_argument('--feature-type', choices=['HAAR', 'LBP', 'HOG'], default=DEFAULT_FEATURE_TYPE)
    parser.add_argument('--max-false-alarm-rate', type=float, default=DEFAULT_MAX_FALSE_ALARM_RATE)
    parser.add_argument('--window-width', type=int, default=WINDOW_SIZE[0])
    parser.add_argument('--window-height', type=int, default=WINDOW_SIZE[1])
    parser.add_argument('--cascade-dir', default=CASCADE_DIR.name)
    parser.add_argument('--clean-output', action='store_true')
    parser.add_argument('--hard-negatives-per-image', type=int, default=6)
    parser.add_argument('--max-source-images', type=int, default=None)
    parser.add_argument('--overwrite-hard-negatives', action='store_true')
    args = parser.parse_args()
    set_training_dataset(args.dataset)
    window_size = (args.window_width, args.window_height)
    cascade_dir = resolve_cascade_dir(args.cascade_dir)

    action_requested = any([
        args.build_apps,
        args.annotate,
        args.generate_neg,
        args.generate_hard_negatives,
        args.validate,
        args.generate_vec,
        args.train,
    ])

    if args.commands or not action_requested:
        print_commands()
    if args.build_apps:
        build_opencv_apps()
    if args.annotate:
        run_annotation()
    if args.generate_neg:
        generate_negative_description_file()
    if args.generate_hard_negatives:
        generate_hard_negative_crops(
            args.hard_negatives_per_image,
            args.max_source_images,
            args.overwrite_hard_negatives,
            window_size,
        )
    if args.validate:
        pos_boxes, neg_count = validate_training_files()
        print(f'Validated training files: {pos_boxes} positive boxes, {neg_count} negative images.')
    if args.generate_vec:
        generate_vec(args.num, window_size)
    if args.train:
        train_cascade(
            args.num_pos,
            args.num_neg,
            args.max_false_alarm_rate,
            args.num_stages,
            args.feature_type,
            cascade_dir,
            window_size,
            args.clean_output,
        )


if __name__ == '__main__':
    main()
