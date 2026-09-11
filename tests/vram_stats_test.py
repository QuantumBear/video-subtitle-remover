import pytest

pytest.importorskip('av')
import vsr_pipeline


def test_cuda_memory_snapshot_reports_unavailable_without_cuda(monkeypatch, capsys):
    monkeypatch.setattr(vsr_pipeline.torch.cuda, 'is_available', lambda: False)

    assert vsr_pipeline.cuda_memory_snapshot('test') is None
    assert '[vram] test: unavailable (cuda=false)' in capsys.readouterr().out


def test_cuda_memory_snapshot_reports_current_peak_and_capacity(monkeypatch, capsys):
    cuda = vsr_pipeline.torch.cuda
    monkeypatch.setattr(cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cuda, 'memory_allocated', lambda: 1 * 1024**3)
    monkeypatch.setattr(cuda, 'memory_reserved', lambda: 2 * 1024**3)
    monkeypatch.setattr(cuda, 'max_memory_allocated', lambda: 3 * 1024**3)
    monkeypatch.setattr(cuda, 'max_memory_reserved', lambda: 4 * 1024**3)
    monkeypatch.setattr(cuda, 'mem_get_info', lambda: (5 * 1024**3, 6 * 1024**3))

    snapshot = vsr_pipeline.cuda_memory_snapshot('test')
    output = capsys.readouterr().out

    assert snapshot['allocated'] == 1 * 1024**3
    assert 'allocated=1.00GiB' in output
    assert 'reserved=2.00GiB' in output
    assert 'peak_allocated=3.00GiB' in output
    assert 'peak_reserved=4.00GiB' in output
    assert 'free=5.00GiB total=6.00GiB' in output


def test_cuda_memory_snapshot_can_reset_peak_stats(monkeypatch):
    cuda = vsr_pipeline.torch.cuda
    monkeypatch.setattr(cuda, 'is_available', lambda: True)
    calls = []
    monkeypatch.setattr(cuda, 'reset_peak_memory_stats', lambda: calls.append(True))
    monkeypatch.setattr(cuda, 'memory_allocated', lambda: 0)
    monkeypatch.setattr(cuda, 'memory_reserved', lambda: 0)
    monkeypatch.setattr(cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(cuda, 'max_memory_reserved', lambda: 0)
    monkeypatch.setattr(cuda, 'mem_get_info', lambda: (0, 1))

    vsr_pipeline.cuda_memory_snapshot('test', reset_peak=True)
    assert calls == [True]
