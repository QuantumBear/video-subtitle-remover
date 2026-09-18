"""STTN 首层 Q/K/V 投影缓存的等价性约束。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("torchvision")

from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("frame_count", [1, 3])
def test_first_attention_qkv_cache_matches_direct_inference(frame_count):
    """缓存只跳过首层投影，不能改变 STTN 的输出。"""
    model = InpaintGenerator(init_weights=False).eval()
    features = torch.randn(frame_count, 256, 60, 108)
    masks = torch.zeros(frame_count, 1, 240, 432)

    with torch.inference_mode():
        direct = model.infer(features, masks)
        cached_qkv = model.project_first_qkv(features)
        cached = model.infer(features, masks, qkv_cache=cached_qkv)

    torch.testing.assert_close(cached, direct, rtol=0, atol=0)


def small_engine():
    """保留真实八层网络和窗口循环，缩小空间分块以控制 CPU 测试成本。"""
    torch.manual_seed(31)
    engine = STTNDetInpaint.__new__(STTNDetInpaint)
    engine.device = torch.device("cpu")
    engine.model_input_width = engine.model_input_height = 32
    engine.neighbor_stride = 2
    engine.ref_length = 3
    engine.cache_first_qkv = True
    engine.model = InpaintGenerator(init_weights=False).eval()
    for block in engine.model.transformer:
        block.attention.patchsize = [(8, 8), (4, 4), (2, 2), (1, 1)]
    return engine


def test_reordered_overlapping_windows_match_uncached_features():
    model = small_engine().model
    features = torch.randn(5, 256, 8, 8)
    masks = torch.zeros(5, 1, 32, 32)
    for i in range(1, 5):
        masks[i, :, 8:24, i:16 + i] = 1
    with torch.inference_mode():
        cache = model.project_first_qkv(features)
        original_cache = tuple(item.clone() for item in cache)
        for ids in ([0, 1, 2, 4], [2, 3, 4, 0], [4]):
            expected = model.infer(features[ids], masks[ids])
            actual = model.infer(features[ids], masks[ids],
                                 qkv_cache=tuple(item[ids] for item in cache))
            # 整段与窗口的卷积 batch 大小不同，允许 FP32 舍入误差。
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for actual, expected in zip(cache, original_cache):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_single_window_does_not_allocate_a_segment_cache(monkeypatch):
    engine = small_engine()

    def unexpected_cache(features):
        pytest.fail("单窗口没有跨窗口复用，不应分配整段缓存")

    monkeypatch.setattr(engine.model, "project_first_qkv", unexpected_cache)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    assert len(engine.inpaint([frame], [mask])) == 1


@pytest.mark.parametrize("frame_count", [1, 7])
def test_engine_reuses_only_first_layer_and_keeps_cache_local(frame_count):
    engine = small_engine()
    counts = {(layer, name): [] for layer in range(8) for name in ("query", "key", "value")}
    handles = []
    for (layer, name), calls in counts.items():
        projection = getattr(engine.model.transformer[layer].attention, name + "_embedding")
        handles.append(projection.register_forward_pre_hook(
            lambda module, inputs, calls=calls: calls.append(inputs[0].shape[0])))
    rng = np.random.default_rng(42)
    try:
        # 第二段更换帧和 mask，避免常驻引擎错误复用前一段缓存。
        for segment in range(2):
            frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                      for _ in range(frame_count)]
            masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
            for i, mask in enumerate(masks):
                if i % 3 != 0:
                    mask[8:24, 4 + segment:20 + segment] = 255

            engine.cache_first_qkv = False
            for calls in counts.values():
                calls.clear()
            expected = engine.inpaint(frames.copy(), masks.copy())
            baseline_counts = {key: calls.copy() for key, calls in counts.items()}

            engine.cache_first_qkv = True
            for calls in counts.values():
                calls.clear()
            actual = engine.inpaint(frames.copy(), masks.copy())
            for key, calls in counts.items():
                if key[0] == 0:
                    assert calls == [frame_count]
                    if frame_count > engine.neighbor_stride:
                        assert sum(baseline_counts[key]) > frame_count
                else:
                    assert calls == baseline_counts[key]
            for output, reference in zip(actual, expected):
                np.testing.assert_allclose(output, reference, rtol=0, atol=1)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('cached', [False, True])
def test_profile_reports_stages_without_changing_output(cached, capsys):
    engine = small_engine()
    engine.cache_first_qkv = cached
    rng = np.random.default_rng(73)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(7)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    masks[2][8:24, 8:24] = 255
    engine.profile = False
    expected = engine.inpaint(frames.copy(), masks.copy())
    assert '[sttn-profile]' not in capsys.readouterr().out
    engine.profile = True
    actual = engine.inpaint(frames.copy(), masks.copy())
    report = engine.last_profile
    assert report['frames'] == 7
    assert report['windows'] == 4
    assert report['input_frame_visits'] > 7
    assert report['decoded_frame_visits'] > 7
    assert report['cache_active'] is cached
    for stage in ('preprocess', 'upload', 'encoder', 'gather', 'transformer',
                  'decoder', 'download', 'postprocess'):
        assert report['stage_seconds'][stage] >= 0
    assert ('qkv' in report['stage_seconds']) is cached
    for output, reference in zip(actual, expected):
        np.testing.assert_array_equal(output, reference)
    log = capsys.readouterr().out
    assert '[sttn-profile]' in log
    assert 'clock=wall' in log
    assert 'frames=7 windows=4' in log
    assert 'download_wait_wall=' in log
    # 同一常驻引擎切换到短段不应累计前一段结果。
    engine.inpaint(frames[:1], masks[:1])
    assert engine.last_profile['frames'] == 1
    assert engine.last_profile['windows'] == 1
    assert not engine.last_profile['cache_active']
