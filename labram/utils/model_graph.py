# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Model-architecture visualization coloured by which layers are frozen vs
# trainable. Primary renderer is Graphviz (vector SVG); matplotlib is a fallback
# so a graph is produced even where the Graphviz `dot` binary is unavailable.
# ---------------------------------------------------------

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from labram.utils.logging import get_logger

logger = get_logger(__name__)

# state -> (fill colour, human label)
_STATE_STYLE = {
    'trainable': ('#8bc34a', 'trainable'),   # green
    'frozen':    ('#e57373', 'frozen'),       # red
    'mixed':     ('#ffb74d', 'mixed'),        # orange
    'none':      ('#e0e0e0', 'no params'),    # grey
}


@dataclass
class GraphNode:
    path: str            # dotted module path ('' for root)
    label: str           # short display name
    type_name: str       # module class name
    depth: int
    parent: Optional[str]
    n_total: int
    n_trainable: int
    state: str


def _param_state(module) -> tuple:
    total = trainable = 0
    for p in module.parameters(recurse=True):
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    if total == 0:
        return 'none', 0, 0
    if trainable == total:
        return 'trainable', total, trainable
    if trainable == 0:
        return 'frozen', total, trainable
    return 'mixed', total, trainable


def collect_nodes(model, max_depth: int = 3) -> List[GraphNode]:
    """Flatten the module tree (down to ``max_depth`` edges from the root) into
    coloured nodes. A node's state reflects all parameters beneath it, so a block
    with any frozen weight shows as ``frozen``/``mixed`` even when collapsed."""
    nodes: List[GraphNode] = []

    def visit(module, path, label, depth, parent):
        state, total, trainable = _param_state(module)
        nodes.append(GraphNode(
            path=path or '<root>', label=label, type_name=type(module).__name__,
            depth=depth, parent=parent, n_total=total, n_trainable=trainable, state=state))
        if depth >= max_depth:
            return
        for child_name, child in module.named_children():
            child_path = f"{path}.{child_name}" if path else child_name
            visit(child, child_path, child_name, depth + 1, path or '<root>')

    visit(model, '', type(model).__name__, 0, None)
    return nodes


def _node_text(n: GraphNode) -> str:
    pct = (100.0 * n.n_trainable / n.n_total) if n.n_total else 0.0
    if n.n_total == 0:
        return f"{n.label}\n({n.type_name})"
    return f"{n.label}\n({n.type_name})\n{n.n_trainable:,}/{n.n_total:,} ({pct:.0f}%)"


def build_dot(model, max_depth: int = 3):
    """Return a ``graphviz.Digraph`` of the model coloured by trainability, or
    ``None`` if the graphviz python package is unavailable."""
    try:
        import graphviz
    except Exception:
        return None
    nodes = collect_nodes(model, max_depth)
    dot = graphviz.Digraph(
        'model',
        graph_attr={'rankdir': 'LR', 'fontsize': '12', 'label': 'Model (green=trainable, red=frozen, orange=mixed)'},
        node_attr={'style': 'filled,rounded', 'shape': 'box', 'fontsize': '10', 'fontname': 'Helvetica'})
    for n in nodes:
        fill, _ = _STATE_STYLE[n.state]
        dot.node(n.path, _node_text(n), fillcolor=fill)
    for n in nodes:
        if n.parent is not None:
            dot.edge(n.parent, n.path)
    return dot


def matplotlib_figure(model, max_depth: int = 3):
    """A colour-coded box-per-layer figure (fallback when Graphviz can't render).
    Returns a matplotlib Figure, or ``None`` if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg', force=False)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return None
    # Show the leaf-most nodes (deepest kept level) for a compact freeze map.
    nodes = [n for n in collect_nodes(model, max_depth) if n.n_total > 0 and n.depth > 0]
    if not nodes:
        nodes = [n for n in collect_nodes(model, max_depth) if n.depth == 0]
    h = max(3, 0.35 * len(nodes) + 1)
    fig, ax = plt.subplots(figsize=(9, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(nodes))
    ax.axis('off')
    for i, n in enumerate(reversed(nodes)):
        fill, _ = _STATE_STYLE[n.state]
        ax.add_patch(plt.Rectangle((0.02, i + 0.1), 0.96, 0.8, facecolor=fill,
                                   edgecolor='#333', linewidth=0.6))
        pct = (100.0 * n.n_trainable / n.n_total) if n.n_total else 0.0
        indent = "  " * (n.depth - 1)
        ax.text(0.04, i + 0.5, f"{indent}{n.path}  ·  {n.type_name}  ·  "
                f"{n.n_trainable:,}/{n.n_total:,} ({pct:.0f}% trainable)",
                va='center', ha='left', fontsize=8)
    ax.set_title('Model layers — green=trainable, red=frozen, orange=mixed', fontsize=10)
    legend = [Patch(facecolor=c, label=lbl) for c, lbl in _STATE_STYLE.values()]
    ax.legend(handles=legend, loc='lower right', fontsize=8, ncol=len(legend))
    fig.tight_layout()
    return fig


def save_model_graph(model, output_dir: str, fmt: str = 'svg',
                     max_depth: int = 3, basename: str = 'model_graph') -> Dict[str, Any]:
    """Render the model graph to ``output_dir``.

    Writes the Graphviz DOT source always, renders it to ``fmt`` (svg/png) when
    the ``dot`` binary is available, and always builds a matplotlib figure as a
    portable fallback. Returns paths + the figure for downstream logging.
    """
    result: Dict[str, Any] = {'image_path': None, 'dot_path': None, 'figure': None, 'format': fmt}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dot = build_dot(model, max_depth)
    if dot is not None and output_dir:
        dot_path = os.path.join(output_dir, f'{basename}.dot')
        try:
            with open(dot_path, 'w', encoding='utf-8') as f:
                f.write(dot.source)
            result['dot_path'] = dot_path
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write model graph DOT source: %s", exc)
        try:
            dot.format = fmt
            rendered = dot.render(os.path.join(output_dir, basename), cleanup=True)
            result['image_path'] = rendered
            logger.info("Rendered model graph -> %s", rendered)
        except Exception as exc:
            logger.warning("Graphviz render failed (%s); falling back to matplotlib.", exc)

    fig = matplotlib_figure(model, max_depth)
    result['figure'] = fig
    if result['image_path'] is None and fig is not None and output_dir:
        img_path = os.path.join(output_dir, f'{basename}.{fmt if fmt in ("svg", "png") else "png"}')
        try:
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            result['image_path'] = img_path
            logger.info("Saved model graph (matplotlib) -> %s", img_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to save matplotlib model graph: %s", exc)
    return result
