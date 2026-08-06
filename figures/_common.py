#!/usr/bin/env python
"""Shared style, palette and I/O for the paper figures.

Everything the fig*.py scripts have in common lives here so that a change to the palette or to the
export settings propagates to every panel instead of being copied five times.

Paths are relative to this file, so the package runs from anywhere once the repository is cloned and
the Zenodo archive is unpacked into data/:

    ARES/
      data/        <- Zenodo archive unpacked here
      figures/     <- this directory
        output/    <- figures are written here

Set the environment variable ARES_DATA to read the data from somewhere else.
"""
import colorsys
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('ARES_DATA', os.path.join(HERE, os.pardir, 'data'))
OUT = os.path.join(HERE, 'output')


def data(*parts):
    """Absolute path to a file in the data archive; fails loudly if it is not there."""
    p = os.path.normpath(os.path.join(DATA, *parts))
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'missing data file: {p}\n'
            f'Unpack the Zenodo archive into {os.path.normpath(DATA)}, '
            f'or set ARES_DATA to its location.')
    return p


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'svg.fonttype': 'none',      # editable text in SVG
    'pdf.fonttype': 42,          # editable TrueType text in PDF
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'legend.frameon': False,
})


def desat(h, sat=0.50, val=1.02):
    r, g, b = mpl.colors.to_rgb(h)
    H, S, V = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(H, S * sat, min(1, V * val))


# cooperation-mode palette, desaturated once here and reused everywhere
_RAW = {'SEQUENCE': '#3b6fb6', 'PROTEIN': '#c0392b', 'CONTEXT': '#2e8b57',
        'UNRESOLVED': '#9aa0a6', 'ARTIFACT': '#7d6b5d'}
CC = {k: desat(v) for k, v in _RAW.items()}
INK = '#36363f'
AXG = '#9a9a9a'


def dk(c, f=0.62):
    """Darker edge colour for a fill."""
    r, g, b = mpl.colors.to_rgb(c)
    return (r * f, g * f, b * f)


def spines(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=INK, labelsize=6.4, width=0.7)


def pbody(p):
    """P-value body in Nature style: scientific below 0.01, else two significant figures.

    Uses mathtext rather than Unicode superscript characters, because Arial on some systems has no
    U+207B (superscript minus) or U+2070/U+2074-2079, which silently renders as missing-glyph boxes.
    """
    import numpy as np
    p = max(float(p), 1e-300)
    if p < 1e-2:
        e = int(np.floor(np.log10(p)))
        m = p / 10 ** e
        mant = f'{m:.0f}' if abs(m - round(m)) < 0.05 else f'{m:.1f}'
        return rf'{mant} $\times$ 10$^{{{e}}}$'
    return f'{p:.2g}'


def save(fig, name, subdir='fig1', dpi=600):
    """Write a vector PDF into figures/output/<subdir>/.

    PDF rather than PNG because the panels are line art: it stays sharp at any size and remains
    editable in Illustrator, which is what the journal asks for. pdf.fonttype 42 above keeps the
    text as real text rather than outlines. Add 'png' to EXT below if a raster preview is wanted.
    """
    EXT = ('pdf',)
    outdir = os.path.join(OUT, subdir)
    os.makedirs(outdir, exist_ok=True)
    for ext in EXT:
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'  saved {subdir}/{name}.pdf')


# mechanism taxonomy: the leaves under each cooperation mode, and their display labels
LEAVES = {
    'SEQUENCE': ['DIRECT_SEQUENCE_RECOGNITION', 'MOTIF_GRAMMAR'],
    'PROTEIN': ['COOPERATIVE_COBINDING', 'TETHERING', 'CO_OCCUPANCY', 'COMPETITIVE_EXCLUSION'],
    'CONTEXT': ['CHROMATIN_STATE_CONTEXT', 'SEQUENCE_FEATURE_PROXY', 'SURROGATE_TF_COOCCUPANCY',
                'HOT_REGION_COOCCUPANCY', 'CHROMATIN_ACCESSIBILITY_PROXY', '3D_LOOP_CONTEXT'],
    'ARTIFACT': ['TECHNICAL_ARTIFACT', 'PARTNER_MOTIF_UNSUPPORTED'],
}
LBL = {
    'DIRECT_SEQUENCE_RECOGNITION': 'Direct sequence recognition', 'MOTIF_GRAMMAR': 'Motif grammar',
    'COOPERATIVE_COBINDING': 'Cooperative co-binding', 'TETHERING': 'Tethering',
    'CO_OCCUPANCY': 'Co-occupancy', 'COMPETITIVE_EXCLUSION': 'Competitive exclusion',
    'CHROMATIN_STATE_CONTEXT': 'Chromatin state', 'SEQUENCE_FEATURE_PROXY': 'Sequence-feature proxy',
    'SURROGATE_TF_COOCCUPANCY': 'Surrogate-TF co-occupancy', 'HOT_REGION_COOCCUPANCY': 'HOT region',
    'CHROMATIN_ACCESSIBILITY_PROXY': 'Accessibility proxy', '3D_LOOP_CONTEXT': '3D loop',
    'TECHNICAL_ARTIFACT': 'Technical artifact', 'PARTNER_MOTIF_UNSUPPORTED': 'Motif unsupported',
}
MODES = ['SEQUENCE', 'PROTEIN', 'CONTEXT']


def atlas():
    """The ARES atlas, with the column names the figure code uses (tf_a / tf_b / consensus /
    reader_cat), while keeping the originals (target / partner / dichotomy / reader_category) too."""
    import pandas as pd
    a = pd.read_csv(data('ares_atlas.tsv'), sep='\t')
    return a.rename(columns={'target': 'tf_a', 'partner': 'tf_b', 'dichotomy': 'consensus',
                             'reader_category': 'reader_cat'})


def with_route(df, cell='cell', target='target', partner='partner', out='consensus',
               mechanism=False):
    """Attach the cooperation route from the atlas, keyed on (cell, target, partner).

    Route and mechanism are read from the atlas here and nowhere else. A per-analysis table that
    also carried its own copy of a label would be a second definition of it, free to disagree with
    the first, so any such column is dropped before the atlas value is attached.

    Set mechanism=True to attach the mechanism leaf as well.
    """
    import pandas as pd
    a = pd.read_csv(data('ares_atlas.tsv'), sep='\t')
    key = ['cell', 'target', 'partner']
    d = df.drop(columns=[c for c in ('consensus', 'dichotomy', 'dichotomy_class', 'mechanism')
                         if c in df.columns and c != out], errors='ignore').copy()
    k = list(zip(d[cell], d[target], d[partner]))
    d[out] = [a.set_index(key).dichotomy.to_dict().get(x) for x in k]
    if mechanism:
        d['mechanism'] = [a.set_index(key).mechanism.to_dict().get(x) for x in k]
    return d
