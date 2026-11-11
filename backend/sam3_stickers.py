"""Sparse DINO emoji support for SAM3 text masks; no additional SAM3 video pass."""
import cv2

DEFAULT_PROMPT = 'emoji. sticker. cartoon icon.'


def plan_samples(total, fps, cuts, budget):
    """Prefer shot starts and endpoints, then cover time without component jitter."""
    if total <= 0 or budget <= 0:
        return []
    step = max(1, round(fps))
    priority = list(dict.fromkeys([0, *(n for n in sorted(cuts) if 0 <= n < total), total - 1]))
    grid = sorted((set(range(0, total, step)) |
                   set(range(0, min(total, 2 * step + 1), max(1, round(fps / 2))))) - set(priority))
    remaining = max(0, budget - len(priority))
    if len(grid) > remaining:
        grid = [grid[min(len(grid) - 1, int((i + .5) * len(grid) / remaining))]
                for i in range(remaining)]
    return sorted(priority[:budget] + grid)


def detection_regions(text_timeline, roi):
    """Expand each current text envelope by twice its height; empty text uses ROI."""
    ry1, ry2, rx1, rx2 = roi
    result = []
    for boxes in text_timeline:
        if not boxes:
            result.append(roi)
            continue
        top, bottom = min(b[0] for b in boxes), max(b[1] for b in boxes)
        left, right = min(b[2] for b in boxes), max(b[3] for b in boxes)
        pad = 2 * (bottom - top)
        result.append((max(ry1, top - pad), min(ry2, bottom + pad),
                       max(rx1, left - pad), min(rx2, right + pad)))
    return result


def merge_masks(store, text_timeline, sticker_boxes, roi, pad):
    """Union accepted boxes into lossless masks and rebuild only affected schedules."""
    ry1, ry2, rx1, rx2 = roi
    frames = added = 0
    for n, boxes in sticker_boxes.items():
        mask = store[n]
        before = cv2.countNonZero(mask)
        for y1, y2, x1, x2 in boxes:
            mask[max(ry1, y1 - pad):min(ry2, y2 + pad),
                 max(rx1, x1 - pad):min(rx2, x2 + pad)] = 255
        delta = cv2.countNonZero(mask) - before
        if not delta:
            continue
        store.replace(n, mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_timeline[n] = [(y, y+h, x, x+w) for c in contours
                            for x, y, w, h in [cv2.boundingRect(c)]]
        frames += 1
        added += delta
    return {'gdino_emoji_frames': frames, 'gdino_emoji_added_pixels': added}
