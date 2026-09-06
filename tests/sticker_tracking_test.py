"""贴纸/emoji 采样框跟踪的回归测试。

测试不初始化 OCR、LAMA 或 ProPainter；只验证 VLM 结果进入时间线前的
多目标跟踪逻辑。服务器完整依赖安装后可直接用 pytest 运行。
"""
import sys
import types

import numpy as np

# 本地开发环境可能没有视频推理依赖；导入 vsr_pipeline 时这些模块只在
# 实际处理视频/调用模型的方法体内使用，测试可用空模块替代。
sys.modules.setdefault("av", types.ModuleType("av"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from vsr_pipeline import (
    MASK_PAD,
    PROPAINTER_OVERLAP,
    PROPAINTER_SEG_LEN,
    PROPAINTER_SUB_VIDEO_LENGTH,
    STICKER_MASK_PAD,
    _group_sticker_boxes,
    _sticker_box_from_vlm,
    _sticker_match_score,
    Pipeline,
)


def test_adjacent_stickers_keep_separate_tracks():
    """同一帧相邻的三个 emoji 必须各自形成稳定轨迹，不能被合成一个框。"""
    hits = {
        0: [
            (490, 525, 312, 342),
            (490, 525, 343, 373),
            (490, 525, 374, 404),
        ],
        40: [
            (491, 526, 313, 343),
            (491, 526, 344, 374),
            (491, 526, 375, 405),
        ],
        80: [
            (490, 525, 312, 342),
            (490, 525, 343, 373),
            (490, 525, 374, 404),
        ],
    }

    stable = _group_sticker_boxes(hits)

    assert len(stable) == 3
    assert all(len(set(frames)) == 3 for _, frames in stable)


def test_adjacent_boxes_do_not_match_even_when_vlm_boxes_touch():
    """VLM 框略大而相邻时，中心距离仍应阻止错误合并。"""
    left = (490, 525, 310, 350)
    right = (490, 525, 340, 380)

    assert _sticker_match_score(left, right) < 0


def test_sticker_coordinate_conversion_uses_wider_padding():
    """贴纸坐标换算应使用独立的 12px 外扩，而非字幕的 4px。"""
    box = _sticker_box_from_vlm((100, 100, 200, 200), (400, 600, 200, 500))

    assert STICKER_MASK_PAD > MASK_PAD
    assert box == (408, 452, 218, 272)


def test_propainter_window_fits_24gb_profile():
    """24GB 配置的单次输入应为 60 帧，且保留 20 帧边界上下文。"""
    assert PROPAINTER_SEG_LEN == 40
    assert PROPAINTER_OVERLAP == 20
    assert PROPAINTER_SUB_VIDEO_LENGTH == 60


def test_propainter_text_mask_preserves_background_between_glyphs():
    """横向白字幕框只遮字形,不能把字间的楼梯/人物一起擦掉。"""
    pipe = Pipeline.__new__(Pipeline)
    # 测试只关注遮罩策略;跳过依赖 OpenCV 的全局连通域过滤。
    pipe.filter_glyph_by_height = lambda glyph: glyph
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    frame[40:60, 20:60] = 255
    frame[40:60, 80:120] = 255

    mask = pipe.propainter_boxes_to_mask(
        [(35, 65, 15, 125)], frame, (0, 120, 0, 240))

    assert mask[50, 30] == 255       # 字形被遮住
    assert mask[50, 70] == 0         # 字形之间的真实背景保留
    assert mask[50, 120] == 0        # OCR 框外背景也保留


def test_propainter_sticker_box_still_uses_full_rectangle():
    """近方形贴纸框没有白字形时仍须整框擦除。"""
    pipe = Pipeline.__new__(Pipeline)
    pipe.filter_glyph_by_height = lambda glyph: glyph
    frame = np.zeros((120, 240, 3), dtype=np.uint8)

    mask = pipe.propainter_boxes_to_mask(
        [(35, 75, 150, 190)], frame, (0, 120, 0, 240))

    assert mask[50, 170] == 255
