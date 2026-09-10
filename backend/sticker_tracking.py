"""Appearance-confirmed sticker masks and budgeted model sampling.

Only strong detections create references. Local matches never refresh them or
count as independent detections. Coordinates are unpadded full-frame yxyx.
"""

from bisect import bisect_right
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.sticker_detect import _box_iou
from backend.subtitle_tracking import sticker_match_score


@dataclass
class _Reference:
    frame: int
    box: tuple
    patch: np.ndarray
    mask: np.ndarray


@dataclass
class _Track:
    scene: int
    references: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)


def _reference(frame_no, rgb, box):
    y1, y2, x1, x2 = box
    patch = rgb[y1:y2, x1:x2].copy()
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    if patch.size and min(patch.shape[:2]) >= 4:
        try:
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        except (cv2.error, ValueError, FloatingPointError):
            return _Reference(frame_no, box, patch, mask)
        mask[(hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)] = 255
        # Saturated foreground tolerates changing scenery behind emoji edges.
        if np.count_nonzero(mask) < max(12, mask.size * 0.1):
            mask[:] = 255
        values = patch[mask > 0].astype(np.float32)
        if np.max(np.std(values, axis=0)) < 10:
            mask[:] = 0
    return _Reference(frame_no, box, patch, mask)


def _match_reference(rgb, reference, expected):
    """Return (confirmed box, unknown); a failed match is evidence of absence."""
    patch, mask = reference.patch, reference.mask
    count = np.count_nonzero(mask)
    if not count:
        return None, True
    height, width = patch.shape[:2]
    ey1, ey2, ex1, ex2 = expected
    y, x = round((ey1 + ey2 - height) / 2), round((ex1 + ex2 - width) / 2)
    radius = max(2, min(8, round(min(height, width) * 0.2)))
    top, left = max(0, y - radius), max(0, x - radius)
    bottom = min(rgb.shape[0], y + height + radius)
    right = min(rgb.shape[1], x + width + radius)
    search = rgb[top:bottom, left:right]
    if search.shape[0] < height or search.shape[1] < width:
        return None, True
    distances = cv2.matchTemplate(search, patch, cv2.TM_SQDIFF, mask=mask)
    distances = np.where(np.isfinite(distances), distances, np.inf)
    row, col = np.unravel_index(np.argmin(distances), distances.shape)
    if not np.isfinite(distances[row, col]):
        return None, True
    # Test the expected position too: changed edge pixels can displace the
    # full-patch SSD minimum even when the foreground has not moved.
    locations = {(int(row), int(col))}
    if 0 <= y - top < distances.shape[0] and 0 <= x - left < distances.shape[1]:
        locations.add((y - top, x - left))
    best = None
    for row, col in sorted(locations):
        actual = search[row:row + height, col:col + width][mask > 0].astype(np.float32)
        target = patch[mask > 0].astype(np.float32)
        # Tolerate at most a small changing background/edge fringe.
        errors = np.mean((actual - target) ** 2, axis=1)
        keep = errors <= np.quantile(errors, 0.8)
        rmse = np.sqrt(np.mean(errors[keep]))
        actual, target = actual[keep], target[keep]
        actual -= actual.mean(axis=0)
        target -= target.mean(axis=0)
        norm = np.linalg.norm(actual) * np.linalg.norm(target)
        correlation = float(np.sum(actual * target) / norm) if norm > 0 else 0
        if rmse <= 32 and correlation >= 0.55 and (best is None or rmse < best[0]):
            y1, x1 = top + row, left + col
            best = rmse, (y1, y1 + height, x1, x1 + width)
    return (best[1] if best else None), False


class StickerTracker:
    def __init__(self, text_timeline, total_frames, score_threshold=0.25,
                 max_gap=60, scene_change_frames=(), single_hit_radius=6,
                 max_distance=120):
        self.text = text_timeline
        self.total = max(0, int(total_frames))
        self.score = float(score_threshold)
        self.max_gap = max(0, int(max_gap))
        self.radius = max(0, int(single_hit_radius))
        self.max_distance = max(0, float(max_distance))
        self.cuts = sorted(set(n for n in scene_change_frames if 0 < n < self.total))
        self.tracks = []
        self._frame_stats = {}
        self._previous = {}
        self._previous_frame = -1
        self.active = False

    def _scene(self, n):
        return bisect_right(self.cuts, n)

    def _nearest_reference(self, track, n):
        nearby = [r for r in track.references.values() if abs(n - r.frame) <= self.max_gap]
        # A failed new patch must not hide a usable reference or renew its age.
        return min(nearby, key=lambda r: (not np.any(r.mask), abs(n - r.frame)), default=None)

    def add_sample(self, n, rgb, candidates):
        if not 0 <= n < self.total:
            return
        strong = [c for c in candidates if c.score >= self.score]
        matches = []
        bridge_matches = set()
        for ci, candidate in enumerate(strong):
            for ti, track in enumerate(self.tracks):
                if track.scene != self._scene(n):
                    continue
                reference = self._nearest_reference(track, n)
                if reference is None:
                    continue
                score = sticker_match_score(candidate.box, reference.box)
                if score >= 0:
                    try:
                        matched, _ = _match_reference(rgb, reference, candidate.box)
                    except (cv2.error, ValueError, FloatingPointError):
                        matched = None
                    if matched is not None or n == reference.frame:
                        matches.append((-score, ti, ci))
                    if matched is not None and _box_iou(matched, candidate.box) >= 0.6:
                        bridge_matches.add((ti, ci))
        used_tracks, used_candidates = set(), set()
        assignments = {}
        for _, ti, ci in sorted(matches):
            if ti in used_tracks or ci in used_candidates:
                continue
            track = self.tracks[ti]
            if n not in track.references:
                track.references[n] = _reference(n, rgb, strong[ci].box)
            used_tracks.add(ti)
            used_candidates.add(ci)
            assignments[ci] = ti
        # A late feedback sample can bridge two temporally disjoint tracks.
        # Both references must confirm this same candidate's local position.
        merged = set()
        for _, ti, ci in sorted(matches):
            target_index = assignments.get(ci)
            if (ti in used_tracks or ti in merged or target_index is None
                    or (ti, ci) not in bridge_matches
                    or (target_index, ci) not in bridge_matches):
                continue
            target, other = self.tracks[target_index], self.tracks[ti]
            if (max(target.references) == n < min(other.references)
                    or max(other.references) < n == min(target.references)):
                target.references.update(other.references)
                target.observations.update(other.observations)
                merged.add(ti)
        if merged:
            retained = [ti for ti in range(len(self.tracks)) if ti not in merged]
            self.tracks = [self.tracks[ti] for ti in retained]
            self._previous = {new: self._previous[old] for new, old in enumerate(retained)
                              if old in self._previous}
        for ci, candidate in enumerate(strong):
            if ci not in used_candidates:
                track = _Track(self._scene(n))
                track.references[n] = _reference(n, rgb, candidate.box)
                self.tracks.append(track)

    def _expected_box(self, track, n):
        refs = list(track.references.values())
        usable = [r for r in refs if np.any(r.mask)]
        refs = sorted(usable or refs, key=lambda r: r.frame)
        right = bisect_right([r.frame for r in refs], n)
        if 0 < right < len(refs):
            before, after = refs[right - 1], refs[right]
            if after.frame - before.frame <= self.max_gap:
                ratio = (n - before.frame) / (after.frame - before.frame)
                return tuple(round(a + (b - a) * ratio)
                             for a, b in zip(before.box, after.box))
        if n - 1 in track.observations:
            return track.observations[n - 1]
        return min(refs, key=lambda r: abs(n - r.frame)).box

    def observe(self, n, rgb, candidates=()):
        counts = dict(weak_continuations=0, appearance_recovered=0,
                      appearance_absent=0, appearance_unknown=0)
        current, eligible = {}, {}
        for ti, track in enumerate(self.tracks):
            if track.scene != self._scene(n):
                continue
            ref = self._nearest_reference(track, n)
            if ref is not None:
                eligible[ti] = ref, self._expected_box(track, n)
        self.active = bool(eligible)
        # Assign candidates one-to-one before local confirmation. A neighboring
        # candidate must never supply evidence to two sticker tracks.
        proposals, used = {}, set()
        pairs = [( -sticker_match_score(c.box, box), ti, ci)
                 for ti, (_, box) in eligible.items() for ci, c in enumerate(candidates)
                 if c.score >= min(0.15, self.score) and sticker_match_score(c.box, box) >= 0]
        for _, ti, ci in sorted(pairs):
            if ti not in proposals and ci not in used:
                proposals[ti] = candidates[ci]
                used.add(ci)
        for ti, (ref, expected) in eligible.items():
            track = self.tracks[ti]
            if n in track.references:
                box = track.references[n].box
            else:
                proposal = proposals.get(ti)
                try:
                    # A weak box cannot move the search to a different target.
                    box, unknown = _match_reference(rgb, ref, expected)
                except (cv2.error, ValueError, FloatingPointError):
                    box, unknown = None, True
                if box is None:
                    counts['appearance_unknown' if unknown else 'appearance_absent'] += 1
                elif proposal is not None and proposal.score < self.score:
                    counts['weak_continuations'] += 1
                else:
                    counts['appearance_recovered'] += 1
            track.observations.pop(n, None)
            if box is not None:
                track.observations[n] = box
                current[ti] = box
        previous = self._previous if self._previous_frame in (n - 1, n) else {}
        changed = set(current) != set(previous) or any(
            max(abs(a - b) for a, b in zip(box, previous[ti])) > 2
            for ti, box in current.items() if ti in previous)
        self._previous = current
        self._previous_frame = n
        self._frame_stats[n] = counts
        return changed

    def _near_text(self, n, box):
        for frame in range(max(0, n - self.radius), min(self.total, n + self.radius + 1)):
            if self._scene(frame) != self._scene(n) or frame >= len(self.text):
                continue
            for text in self.text[frame]:
                dy = max(0, box[0] - text[1], text[0] - box[1])
                dx = max(0, box[2] - text[3], text[2] - box[3])
                if np.hypot(dx, dy) <= self.max_distance:
                    return True
        return False

    def finish(self):
        result = {}
        for track in self.tracks:
            singleton = len(track.references) == 1
            first = min(track.references)
            if singleton and not self._near_text(first, track.references[first].box):
                continue
            for n, box in track.observations.items():
                if singleton and (abs(n - first) > self.radius or not self._near_text(n, box)):
                    continue
                boxes = result.setdefault(n, [])
                if not any(_box_iou(box, previous) >= 0.6 for previous in boxes):
                    boxes.append(box)
        return result

    @property
    def stats(self):
        counts = dict(strong_hits=sum(len(t.references) for t in self.tracks),
                      weak_continuations=0, appearance_recovered=0,
                      appearance_absent=0, appearance_unknown=0)
        for frame_counts in self._frame_stats.values():
            for key, value in frame_counts.items():
                counts[key] += value
        return counts


class FeedbackSchedule:
    def __init__(self, priority_frames, max_calls, max_step=30):
        self.max_calls = max(0, int(max_calls))
        self.priority_frames = list(dict.fromkeys(
            int(n) for n in priority_frames if int(n) >= 0))[:self.max_calls]
        self._priority = set(self.priority_frames)
        self._attempted = set()
        self.max_step = max(1, int(max_step))
        self.step = 1
        self.next_check = 0

    @property
    def calls(self):
        return len(self._attempted)

    def record_priority(self, n):
        self._attempted.add(n)

    def should_check(self, n, changed=False, active=False):
        remaining_priority = len(self._priority - self._attempted)
        return (n not in self._attempted and n not in self._priority
                and self.calls + remaining_priority < self.max_calls
                and active and (changed or n >= self.next_check))

    def record_feedback(self, n, changed=False):
        self._attempted.add(n)
        self.step = 1 if changed else min(self.max_step, self.step * 2)
        self.next_check = n + self.step


def locate_tracked_stickers(video_path, region, sample_frames, detector,
                            text_timeline, total_frames, *, max_calls=200,
                            prompt=None, score_threshold=0.25, max_area_px=1200,
                            scene_change_frames=(), max_gap=60, base_step=30):
    """Reserve priority calls, then confirm local presence and spend spare calls.

    The first decode retains only small reference patches, never the full video.
    A second decode checks skipped frames locally; model failures consume budget
    but are not treated as confirmed absences.
    """
    import os

    import av

    from backend.sticker_detect import DEFAULT_PROMPT

    schedule = FeedbackSchedule((n for n in sample_frames if n < total_frames),
                                max_calls, base_step)
    tracker = StickerTracker(text_timeline, total_frames, score_threshold,
                             max_gap, scene_change_frames)
    samples = {}
    model_failed = budget_skipped = 0
    ymin, ymax, xmin, xmax = region

    def detect(n, rgb):
        nonlocal model_failed
        try:
            candidates = detector.detect_candidates(
                rgb[ymin:ymax, xmin:xmax], region, prompt or DEFAULT_PROMPT,
                score_threshold, max_area_px)
        except Exception as exc:
            model_failed += 1
            print(f'[sticker-gdino] f{n} 检测未确认: {type(exc).__name__}')
            return
        samples[n] = candidates
        tracker.add_sample(n, rgb, candidates)

    priority = set(schedule.priority_frames)
    if priority and total_frames > 0:
        with av.open(os.fspath(video_path)) as src:
            for n, frame in enumerate(src.decode(video=0)):
                if n > max(priority):
                    break
                if n in priority:
                    schedule.record_priority(n)
                    detect(n, frame.to_ndarray(format='rgb24'))
        with av.open(os.fspath(video_path)) as src:
            for n, frame in enumerate(src.decode(video=0)):
                if n >= total_frames:
                    break
                rgb = frame.to_ndarray(format='rgb24')
                changed = tracker.observe(n, rgb, samples.get(n, ()))
                if schedule.should_check(n, changed=changed, active=tracker.active):
                    before = tracker.stats['strong_hits']
                    detect(n, rgb)
                    updated = tracker.observe(n, rgb, samples.get(n, ()))
                    schedule.record_feedback(n, changed=changed or updated and
                                             tracker.stats['strong_hits'] > before)
                elif (n not in priority and tracker.active and
                      (changed or n >= schedule.next_check) and schedule.calls >= schedule.max_calls):
                    budget_skipped += 1
        # New high-score feedback references can confirm earlier frames too.
        # Re-decode instead of retaining full-resolution video in memory.
        if schedule._attempted - priority:
            for track in tracker.tracks:
                track.observations.clear()
            with av.open(os.fspath(video_path)) as src:
                for n, frame in enumerate(src.decode(video=0)):
                    if n >= total_frames:
                        break
                    tracker.observe(n, frame.to_ndarray(format='rgb24'), samples.get(n, ()))
    result = tracker.finish()
    stats = dict(tracker.stats, calls=schedule.calls,
                 priority_calls=len(priority & schedule._attempted),
                 feedback_calls=len(schedule._attempted - priority),
                 model_failed=model_failed, budget_skipped=budget_skipped)
    print(f'[sticker-gdino] calls={stats["calls"]}/{schedule.max_calls} '
          f'priority={stats["priority_calls"]} feedback={stats["feedback_calls"]} '
          f'strong={stats["strong_hits"]} weak={stats["weak_continuations"]} '
          f'local={stats["appearance_recovered"]} absent={stats["appearance_absent"]} '
          f'unknown={stats["appearance_unknown"]} failed={model_failed} '
          f'budget_skipped={budget_skipped}')
    return result, stats
