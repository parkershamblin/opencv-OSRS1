import cv2 as cv
import numpy as np


class Vision:
    def filter_rectangles(
        self,
        rectangles,
        min_size=(55, 40),
        aspect_ratio_range=(1.1, 2.2),
        overlap_threshold=0.35,
    ):
        filtered = []

        for (x, y, w, h) in rectangles:
            if w < min_size[0] or h < min_size[1]:
                continue

            aspect_ratio = w / float(h)
            if not aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                continue

            filtered.append((int(x), int(y), int(w), int(h)))

        return self.non_max_suppression(filtered, overlap_threshold)

    def non_max_suppression(self, rectangles, overlap_threshold=0.35):
        if len(rectangles) == 0:
            return []

        boxes = np.asarray(rectangles, dtype=np.float32)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        area = boxes[:, 2] * boxes[:, 3]
        order = area.argsort()
        picked = []

        while len(order) > 0:
            i = order[-1]
            picked.append(i)
            order = order[:-1]

            if len(order) == 0:
                break

            xx1 = np.maximum(x1[i], x1[order])
            yy1 = np.maximum(y1[i], y1[order])
            xx2 = np.minimum(x2[i], x2[order])
            yy2 = np.minimum(y2[i], y2[order])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / area[order]
            order = order[overlap <= overlap_threshold]

        return boxes[picked].astype(np.int32).tolist()

    # given a list of [x, y, w, h] rectangles returned by find(), convert those into a list of
    # [x, y] positions in the center of those rectangles where we can click on those found items
    def get_click_points(self, rectangles):
        points = []

        # Loop over all the rectangles
        for (x, y, w, h) in rectangles:
            # Determine the center position
            center_x = x + int(w/2)
            center_y = y + int(h/2)
            # Save the points
            points.append((center_x, center_y))

        return points

    # given a list of [x, y, w, h] rectangles and a canvas image to draw on, return an image with
    # all of those rectangles drawn
    def draw_rectangles(self, haystack_img, rectangles):
        # these colors are actually BGR
        line_color = (0, 255, 0)
        line_type = cv.LINE_4

        for (x, y, w, h) in rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box
            cv.rectangle(haystack_img, top_left, bottom_right, line_color, lineType=line_type)

        return haystack_img

    # given a list of [x, y] positions and a canvas image to draw on, return an image with all
    # of those click points drawn on as crosshairs
    def draw_crosshairs(self, haystack_img, points):
        # these colors are actually BGR
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS

        for (center_x, center_y) in points:
            # draw the center point
            cv.drawMarker(haystack_img, (center_x, center_y), marker_color, marker_type)

        return haystack_img

    def centeroid(self, point_list):
        point_list = np.asarray(point_list, dtype=np.int32)
        length = point_list.shape[0]
        sum_x = np.sum(point_list[:, 0])
        sum_y = np.sum(point_list[:, 1])
        return [np.floor_divide(sum_x, length), np.floor_divide(sum_y, length)]
