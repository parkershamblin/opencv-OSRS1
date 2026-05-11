import numpy as np
import win32api, win32gui, win32ui, win32con
from typing import Any


_AUTO_WINDOW: Any = object()
CAPTUREBLT = 0x40000000


class WindowCapture:

    RUNELITE_TITLE_PREFIX = 'RuneLite'
    OSRS_WINDOW_TITLE = 'Old School RuneScape'
    TOP_EDGE_ARTIFACT_MAX_ROWS = 12
    TOP_EDGE_ARTIFACT_MIN_BRIGHT_RATIO = 0.65
    TOP_EDGE_ARTIFACT_MIN_LUMA = 170
    TOP_EDGE_ARTIFACT_MAX_CHANNEL_SPREAD = 55

    # properties
    w = 0
    h = 0
    hwnd: int | None = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0
    window_name = None

    # constructor
    def __init__(self, window_name=_AUTO_WINDOW, capture_from_screen=False):
        # find the handle for the window we want to capture.
        # by default, prefer RuneLite windows and fall back to the official client
        self._auto_find_window = (
            window_name is _AUTO_WINDOW or window_name == self.OSRS_WINDOW_TITLE
        )
        self._capture_desktop = window_name is None
        self._capture_from_screen = capture_from_screen
        self._last_top_trim = 0

        if window_name is _AUTO_WINDOW or window_name == self.OSRS_WINDOW_TITLE:
            self.hwnd, self.window_name = self.find_osrs_window()
        # if None is given explicitly, capture the entire screen
        elif window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
            self.window_name = window_name
            self._capture_from_screen = True
        else:
            hwnd = win32gui.FindWindow(None, window_name)
            if not hwnd:
                raise Exception('Window not found: {}'.format(window_name))
            # store window name for potential re-finding
            self.hwnd = hwnd
            self.window_name = window_name
        
        # initialize window position (will be called again by refresh)
        self._update_window_position()

    @classmethod
    def find_osrs_window(cls):
        """
        Find the preferred OSRS client window.
        Tries any visible title starting with "RuneLite" first, then falls back
        to the exact official client title.
        """
        hwnd, title = cls._find_window_by_title_prefix(cls.RUNELITE_TITLE_PREFIX)
        if hwnd:
            return hwnd, title

        hwnd = win32gui.FindWindow(None, cls.OSRS_WINDOW_TITLE)
        if hwnd:
            return hwnd, cls.OSRS_WINDOW_TITLE

        message = (
            'Neither window found: no window title starting with '
            f'"{cls.RUNELITE_TITLE_PREFIX}" and no window titled '
            f'"{cls.OSRS_WINDOW_TITLE}". Please open RuneLite or '
            'Old School RuneScape and try again.'
        )
        print(message)
        raise Exception(message)

    @staticmethod
    def _find_window_by_title_prefix(title_prefix):
        matches = []

        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title.startswith(title_prefix):
                    ctx.append((hwnd, title))

        win32gui.EnumWindows(winEnumHandler, matches)
        if matches:
            return matches[0]
        return None, None

    def _update_window_position(self):
        """
        Internal method to update window position and size.
        Handles multi-monitor setups by properly accounting for negative coordinates.
        This is called in __init__ and can be called again if window moves.
        """
        if self._capture_desktop:
            self.offset_x = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            self.offset_y = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
            self.w = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            self.h = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            self.cropped_x = 0
            self.cropped_y = 0
            return

        self._ensure_window_handle()

        hwnd = self.hwnd
        if hwnd is None:
            raise Exception('Window handle is unavailable.')

        if win32gui.IsIconic(hwnd):
            raise Exception('Window is minimized. Restore it before capturing.')

        client_rect = win32gui.GetClientRect(hwnd)
        self.w = client_rect[2] - client_rect[0]
        self.h = client_rect[3] - client_rect[1]

        if self.w <= 0 or self.h <= 0:
            raise Exception(f'Window client area has invalid size: {self.w}x{self.h}')

        # GetClientRect/ClientToScreen follows the real client area in both
        # decorated windowed mode and borderless/fullscreen plugin modes.
        self.cropped_x = 0
        self.cropped_y = 0

        # set the client coordinates offset so we can translate screenshot
        # images into actual screen positions.
        # This correctly handles negative coordinates on multi-monitor setups.
        self.offset_x, self.offset_y = win32gui.ClientToScreen(
            hwnd, (client_rect[0], client_rect[1])
        )

    def _ensure_window_handle(self):
        if self.hwnd and win32gui.IsWindow(self.hwnd) and win32gui.IsWindowVisible(self.hwnd):
            return

        if self._auto_find_window:
            self.hwnd, self.window_name = self.find_osrs_window()
            return

        if self.window_name is not None:
            self.hwnd = win32gui.FindWindow(None, self.window_name)
            if self.hwnd:
                return

        raise Exception('Window not found or no longer visible: {}'.format(self.window_name))

    def refresh_window_position(self):
        """
        Refresh the window position. Call this if the window moves between monitors
        or if you're concerned about multi-monitor coordinate drift.
        Safe to call frequently (e.g., every 60 frames) with minimal overhead.
        """
        self._update_window_position()


    def _grab_bitmap_bits(self, capture_from_screen):
        if capture_from_screen:
            wDC = win32gui.GetDC(0)
            release_hwnd = 0
            source_pos = (self.offset_x, self.offset_y)
        else:
            wDC = win32gui.GetDC(self.hwnd)
            release_hwnd = self.hwnd
            source_pos = (0, 0)

        if not wDC:
            raise Exception('Unable to get a device context for screenshot capture')

        dcObj = None
        cDC = None
        dataBitMap = None
        try:
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
            cDC.SelectObject(dataBitMap)
            cDC.BitBlt(
                (0, 0),
                (self.w, self.h),
                dcObj,
                source_pos,
                win32con.SRCCOPY | CAPTUREBLT
            )

            # convert the raw data into a format opencv can read
            #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
            signedIntsArray = dataBitMap.GetBitmapBits(True)
        finally:
            if cDC is not None:
                cDC.DeleteDC()
            if dcObj is not None:
                dcObj.DeleteDC()
            win32gui.ReleaseDC(release_hwnd, wDC)
            if dataBitMap is not None:
                win32gui.DeleteObject(dataBitMap.GetHandle())

        return signedIntsArray

    def get_screenshot(self):
        # Update every frame so fullscreen/windowed transitions and stretched
        # client-size changes are handled immediately.
        self._update_window_position()

        try:
            signedIntsArray = self._grab_bitmap_bits(self._capture_from_screen)
        except Exception:
            if self._capture_from_screen:
                raise
            signedIntsArray = self._grab_bitmap_bits(True)

        # GetBitmapBits returns a bytes-like object; use frombuffer instead of
        # the deprecated fromstring. Specify dtype as a numpy dtype.
        img = np.frombuffer(signedIntsArray, dtype=np.uint8)
        img.shape = (self.h, self.w, 4)

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        top_trim = self._find_top_resize_artifact_height(img)
        self._last_top_trim = top_trim
        if top_trim:
            img = img[top_trim:, :, :]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img

    @classmethod
    def _find_top_resize_artifact_height(cls, img):
        """
        Detect the thin bright strip Windows/RuneLite can expose while resizing.
        Only the top edge is scanned, so this stays cheap in the hot path.
        """
        max_rows = min(cls.TOP_EDGE_ARTIFACT_MAX_ROWS, img.shape[0] - 1)
        trim_height = 0

        for y in range(max_rows):
            row = img[y].astype(np.int16)
            luma = row.mean(axis=1)
            channel_spread = row.max(axis=1) - row.min(axis=1)
            bright_neutral_pixels = (
                (luma >= cls.TOP_EDGE_ARTIFACT_MIN_LUMA) &
                (channel_spread <= cls.TOP_EDGE_ARTIFACT_MAX_CHANNEL_SPREAD)
            )
            bright_ratio = np.mean(bright_neutral_pixels)

            if bright_ratio >= cls.TOP_EDGE_ARTIFACT_MIN_BRIGHT_RATIO:
                trim_height = y + 1
            else:
                break

        return trim_height

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    def get_screen_position(self, pos):
        """
        Translate a pixel position on a screenshot image to a pixel position on the screen.
        
        Args:
            pos: tuple of (x, y) coordinates from the screenshot image
            
        Returns:
            tuple of (screen_x, screen_y) in screen space
            
        Notes:
            - On multi-monitor setups, screen coordinates can be negative if the monitor
              is positioned to the left of the primary monitor. This method correctly
              handles negative coordinates.
            - If the window moves between monitors during execution, call refresh_window_position()
              to update offsets, or it will return incorrect coordinates.
        """
        return (pos[0] + self.offset_x, pos[1] + self.offset_y + self._last_top_trim)
