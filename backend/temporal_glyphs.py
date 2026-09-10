"""Window-local recovery of stationary white captions from original pixels."""

import cv2
import numpy as np


class TemporalGlyphs:
    """Add verified glyphs, never row rectangles or cached repaired pixels.

    Each call is one contiguous same-scene window. Geometric line association
    only proposes a neighborhood; original stroke evidence authorizes edits.
    Unsupported pixels retain their original masks; animated captions may
    only recover a stable subset, not a complete moving text shape.
    """

    MAX_FRAMES = 60
    MAX_SAMPLES = 16

    def __init__(self, region, outline_radius=3):
        self.region = tuple(int(v) for v in region)
        if len(self.region) != 4 or not 0 <= outline_radius <= 4:
            raise ValueError('invalid region or glyph outline radius')
        self.outline_radius = int(outline_radius)

    def _clip(self, box, shape):
        y1, y2, x1, x2 = map(int, box)
        ry1, ry2, rx1, rx2 = self.region
        y1, y2 = max(0, ry1, y1), min(shape[0], ry2, y2)
        x1, x2 = max(0, rx1, x1), min(shape[1], rx2, x2)
        return (y1, y2, x1, x2) if y1 < y2 and x1 < x2 else None

    @staticmethod
    def _same_line(a, b):
        ay1, ay2, ax1, ax2 = a
        by1, by2, bx1, bx2 = b
        ah, bh = ay2 - ay1, by2 - by1
        return (0.65 <= ah / bh <= 1.55
                and abs((ay1 + ay2) - (by1 + by2)) <= 0.5 * min(ah, bh)
                and min(ax2, bx2) - max(ax1, bx1) >= 0.5 * min(ax2 - ax1, bx2 - bx1))

    def _runs(self, boxes, numbers):
        runs, active = [], []
        for i, frame_boxes in enumerate(boxes):
            previous = active if i and numbers[i] == numbers[i - 1] + 1 else []
            active, used = [], set()
            for j, box in enumerate(frame_boxes):
                choices = [r for r in previous if r not in used
                           and self._same_line(runs[r][-1][2], box)]
                if choices:
                    r = min(choices, key=lambda r: abs(sum(runs[r][-1][2][:2]) - sum(box[:2])))
                    runs[r].append((i, j, box))
                    used.add(r)
                else:
                    r = len(runs)
                    runs.append([(i, j, box)])
                active.append(r)
        return [run for run in runs if len(run) >= 3]

    @staticmethod
    def _contrast(gray):
        value = gray.astype(np.float32)
        return value - cv2.GaussianBlur(value, (0, 0), 1)

    @staticmethod
    def _visible(component, expected_contrast, target):
        minimum = target.min(axis=2)
        if np.mean(minimum[component] > 205) < 0.95:
            return False
        observed = TemporalGlyphs._contrast(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
        # Moving background around a fixed stroke is not a caption change.
        # Compare the stroke itself, retaining a per-pixel polarity check.
        expected, actual = expected_contrast[component], observed[component]
        energy = float(np.sum(expected * expected) * np.sum(actual * actual))
        if energy <= 0 or float(np.sum(expected * actual)) / np.sqrt(energy) < 0.7:
            return False
        bright_edges = component & (expected_contrast > 5)
        return (np.count_nonzero(bright_edges) >= 3
                and np.mean(observed[bright_edges] > 0.5) >= 0.8)

    def _recover_run(self, run, frames, masks, new_masks, new_boxes, exclusions):
        shape = masks[0].shape
        height = int(np.median([box[1] - box[0] for _, _, box in run]))
        pad = min(20, max(4, height // 2))
        bounds = self._clip((min(b[0] for _, _, b in run) - 4,
                             max(b[1] for _, _, b in run) + 4,
                             min(b[2] for _, _, b in run) - pad,
                             max(b[3] for _, _, b in run) + pad), shape)
        y1, y2, x1, x2 = bounds
        selected = [run[j] for j in np.linspace(0, len(run) - 1,
                                                min(len(run), self.MAX_SAMPLES), dtype=int)]
        patches = [frames[i][y1:y2, x1:x2] for i, _, _ in selected]
        minimum = np.minimum.reduce([patch.min(axis=2) for patch in patches])
        neutral = np.logical_and.reduce([np.ptp(patch, axis=2) < 40 for patch in patches])
        excluded = np.zeros(minimum.shape, np.uint8)
        for i, _, _ in run:
            for box in exclusions[i]:
                clipped = self._clip(box, shape)
                if clipped is not None:
                    ey1, ey2, ex1, ex2 = clipped
                    if max(ey1, y1) < min(ey2, y2) and max(ex1, x1) < min(ex2, x2):
                        excluded[max(ey1-y1, 0):min(ey2-y1, y2-y1),
                                 max(ex1-x1, 0):min(ex2-x1, x2-x1)] = 255
        core = ((minimum > 215) & neutral & (excluded == 0)).astype(np.uint8)
        support = np.logical_or.reduce([masks[i][y1:y2, x1:x2] > 0 for i, _, _ in selected])
        count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        components = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if (area < 4 or h > height or x == 0 or y == 0
                    or x + w == core.shape[1] or y + h == core.shape[0]):
                continue
            component = labels == label
            distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
            if distance.max() > max(3, height * 0.15):
                continue
            components.append((component, np.count_nonzero(component & support) >= area * 0.5))
        if sum(np.count_nonzero(c) for c, supported in components if supported) < 48:
            return
        # A nearby persistent bullet can be absent from every shortened OCR box.
        supported_pixels = np.logical_or.reduce([c for c, supported in components if supported])
        nearby = cv2.dilate(supported_pixels.astype(np.uint8), np.ones((3, 2 * pad + 1), np.uint8))
        first_supported_x = np.nonzero(supported_pixels)[1].min()
        expected_contrast = self._contrast(minimum)
        outline = np.ones((2 * self.outline_radius + 1,) * 2, np.uint8)
        loose = ((minimum > 165) & (excluded == 0)).astype(np.uint8)
        edge_count, edge_labels, edge_stats, _ = cv2.connectedComponentsWithStats(loose, connectivity=8)
        valid_edges = np.zeros(edge_count, dtype=bool)
        for label in range(1, edge_count):
            x, y, w, h, _ = edge_stats[label]
            valid_edges[label] = (h <= height and x > 0 and y > 0
                                 and x + w < loose.shape[1] and y + h < loose.shape[0])
        for component, supported in components:
            if not supported:
                ys, xs = np.nonzero(component)
                w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
                if (not np.any(component & (nearby > 0)) or xs.max() >= first_supported_x
                        or not 0.65 <= w / h <= 1.5 or max(w, h) > max(4, height * 0.35)
                        or len(xs) < 0.45 * w * h):
                    continue
            near_core = cv2.dilate(component.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            connected = np.zeros(edge_count, dtype=bool)
            connected[np.unique(edge_labels[component])] = True
            connected &= valid_edges
            # Keep core even if its loose edge joins a large bright object.
            # Detached/rejected white components must not return via dilation.
            edge = component | (near_core & connected[edge_labels])
            glyph = cv2.dilate(edge.astype(np.uint8) * 255, outline)
            glyph[excluded > 0] = 0
            for i, j, box in run:
                if not self._visible(component, expected_contrast, frames[i][y1:y2, x1:x2]):
                    continue
                target = new_masks[i][y1:y2, x1:x2]
                added = (glyph > 0) & (target == 0)
                if not added.any():
                    continue
                target[:] = np.maximum(target, glyph)
                ys, xs = np.nonzero(added)
                previous = new_boxes[i][j]
                new_boxes[i][j] = (min(previous[0], y1 + int(ys.min())),
                                    max(previous[1], y1 + int(ys.max()) + 1),
                                    min(previous[2], x1 + int(xs.min())),
                                    max(previous[3], x1 + int(xs.max()) + 1))

    def refine(self, frames_bgr, masks, boxes, frame_numbers, excluded_boxes=None):
        length = len(frames_bgr)
        exclusions = excluded_boxes if excluded_boxes is not None else [[] for _ in frames_bgr]
        if not (length == len(masks) == len(boxes) == len(frame_numbers) == len(exclusions)):
            raise ValueError('frames, masks, boxes, frame numbers and exclusions must align')
        if length > self.MAX_FRAMES:
            raise ValueError('temporal glyph window exceeds 60 frames')
        if not length:
            return [], []
        shape = masks[0].shape
        if any(m.shape != shape or f.shape != (*shape, 3) or m.dtype != np.uint8
               or f.dtype != np.uint8 for f, m in zip(frames_bgr, masks)):
            raise ValueError('expected equal-sized uint8 BGR frames and masks')
        new_masks, new_boxes = [], []
        region = self._clip(self.region, shape)
        for mask, frame_boxes in zip(masks, boxes):
            clipped = np.zeros_like(mask)
            if region:
                y1, y2, x1, x2 = region
                clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            new_masks.append(clipped)
            new_boxes.append([b for box in frame_boxes if (b := self._clip(box, shape)) is not None])
        numbers = [int(n) for n in frame_numbers]
        if len(set(numbers)) != length:
            return new_masks, new_boxes
        for run in self._runs(new_boxes, numbers):
            self._recover_run(run, frames_bgr, masks, new_masks, new_boxes, exclusions)
        return new_masks, new_boxes
