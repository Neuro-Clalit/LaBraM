"""Tests for the model-architecture graph coloured by frozen/trainable layers
(labram.utils.model_graph) and its logging wiring."""

import torch

from labram.utils import model_graph
from labram.utils.model_graph import _STATE_STYLE, collect_nodes, save_model_graph


def _model():
    m = torch.nn.Sequential(
        torch.nn.Linear(4, 4),   # will be frozen
        torch.nn.ReLU(),         # no params
        torch.nn.Linear(4, 2),   # trainable
    )
    for p in m[0].parameters():
        p.requires_grad_(False)
    return m


def test_collect_nodes_states():
    nodes = collect_nodes(_model(), max_depth=2)
    by_path = {n.path: n for n in nodes}
    # child '0' fully frozen, '2' trainable, '1' has no params
    assert by_path['0'].state == 'frozen'
    assert by_path['2'].state == 'trainable'
    assert by_path['1'].state == 'none'
    # root has both frozen and trainable params -> mixed
    assert by_path['<root>'].state == 'mixed'


def test_state_style_covers_all_states():
    for state in ('trainable', 'frozen', 'mixed', 'none'):
        assert state in _STATE_STYLE


def test_save_model_graph_svg_and_dot(tmp_path):
    out = save_model_graph(_model(), str(tmp_path), fmt='svg')
    # graphviz is installed in the test env -> real SVG + DOT source
    assert out['dot_path'] and out['dot_path'].endswith('.dot')
    assert out['image_path'] and out['image_path'].endswith('.svg')
    assert (tmp_path / 'model_graph.svg').stat().st_size > 0
    # DOT source contains the state fill colours
    dot_src = (tmp_path / 'model_graph.dot').read_text()
    assert _STATE_STYLE['frozen'][0] in dot_src
    assert out['figure'] is not None  # matplotlib fallback figure always built


def test_falls_back_to_matplotlib_when_graphviz_unavailable(tmp_path, monkeypatch):
    # Simulate no graphviz python package / dot binary.
    monkeypatch.setattr(model_graph, 'build_dot', lambda *a, **k: None)
    out = save_model_graph(_model(), str(tmp_path), fmt='png')
    assert out['dot_path'] is None
    assert out['image_path'] and out['image_path'].endswith('.png')
    assert (tmp_path / 'model_graph.png').stat().st_size > 0
    assert out['figure'] is not None


class _FakeTask:
    def __init__(self):
        self.artifacts = {}

    def upload_artifact(self, name, artifact_object):
        self.artifacts[name] = artifact_object


class _RecordingWriter:
    def __init__(self, task):
        self.task = task
        self.figures = []
        self.media = []

    def report_figure(self, title, figure, step=None, series="figure"):
        self.figures.append((title, series))

    def report_media(self, title, series, local_path, step=None):
        self.media.append((title, series, local_path))


def test_log_model_visualization_wiring(tmp_path, monkeypatch):
    import labram.runs.common as common
    from labram.configs.train_config import LoggingConfig, OutputConfig
    monkeypatch.setattr(common.utils, "is_main_process", lambda: True)

    task = _FakeTask()
    writer = _RecordingWriter(task)
    common.log_model_visualization(
        LoggingConfig(log_model_graph=True, model_graph_format="svg"),
        OutputConfig(output_dir=str(tmp_path)), _model(), writer)

    assert writer.figures == [("model", "architecture")]      # TensorBoard figure
    assert writer.media and writer.media[0][2].endswith(".svg")  # ClearML vector media
    assert "model_graph" in task.artifacts                    # SVG artifact
    assert "model_graph_dot" in task.artifacts                # DOT source artifact


def test_log_model_visualization_disabled(tmp_path, monkeypatch):
    import labram.runs.common as common
    from labram.configs.train_config import LoggingConfig, OutputConfig
    monkeypatch.setattr(common.utils, "is_main_process", lambda: True)
    writer = _RecordingWriter(_FakeTask())
    common.log_model_visualization(
        LoggingConfig(log_model_graph=False),
        OutputConfig(output_dir=str(tmp_path)), _model(), writer)
    assert writer.figures == [] and writer.media == []
