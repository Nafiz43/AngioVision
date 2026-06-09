"""
========================================================
  SLR Venn Diagram – fully self-contained, no file input
  Edit the DATA SECTION below to update any content.
  Run:  python3 venn_diagram.py
  Output: venn_diagram.pdf  +  venn_diagram.png
========================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

CLUSTERS = {
    'C1': {
        'name'  : 'Data Infrastructure',
        'gaps'  : [
            ('G1',  'Absence of public benchmarks'),
            ('G2',  'Generalizability & domain shift'),
            ('G10', 'Privacy-preserving & multi-institutional learning'),
        ],
    },
    'C2': {
        'name'  : 'Modeling & Learning',
        'gaps'  : [
            ('G3',  'Temporal modeling & video-level analysis'),
            ('G4',  '2D constraints & 3D reconstruction'),
            ('G5',  'Annotation scarcity'),
            ('G9',  'Class imbalance & rare pathology coverage'),
        ],
    },
    'C3': {
        'name'  : 'Clinical Deployment',
        'gaps'  : [
            ('G6',  'clinical validation'),
            ('G7',  'Explainability & model transparency'),
            ('G8',  'Multi-modal data integration'),
        ],
    },
    'C4': {
        'name'  : 'Outcome & Reporting',
        'gaps'  : [
            ('G11', 'clinical outcome prediction'),
            ('G12', 'LLMs & structured reporting'),
        ],
    },
}

STRATEGIES = [
    ('S1',  'Large-scale multi-institutional\nannotated angiographic datasets'),
    ('S2',  'Dedicated challenge benchmarks\nwith standardized evaluation metrics'),
    ('S3',  'Self-supervised / foundation model\npretraining on unlabeled fluoroscopic video'),
    ('S4',  'Federated learning with data-use\nagreements & audit mechanisms'),
    ('S5',  'Domain adaptation for contrast,\ndose, frame rate, and motion'),
    ('S6',  'Video Transformer architectures\nwith long-range attention & SSMs'),
    ('S7',  'Physics-informed networks with\nprojection geometry & multi-view fusion'),
    ('S8',  'Semi-supervised, active & self-\nsupervised learning for annotation'),
    ('S9',  'Uncertainty quantification &\nOOD failure detection'),
    ('S10', 'Prospective multi-center RCTs\nwith clinical outcome endpoints'),
    ('S11', 'Outcome-supervised training for\nclinically meaningful prediction'),
    ('S12', 'Longitudinal cohort datasets linking\nimaging to clinical endpoints'),
    ('S13', 'Vision-language models for image\n& free-text report integration'),
    ('S14', 'Generative augmentation &\nrare-event simulation pipelines'),
]

INTERSECTIONS = {
    frozenset(['C1', 'C2'])             : 'S1',
    frozenset(['C2', 'C3'])             : 'S8, S9',
    frozenset(['C1', 'C4'])             : '—',
    frozenset(['C3', 'C4'])             : 'S10, S11,\nS12, S13',
    frozenset(['C1', 'C3'])             : '—',
    frozenset(['C2', 'C4'])             : '—',
    frozenset(['C1', 'C2', 'C3'])       : '—',
    frozenset(['C1', 'C2', 'C4'])       : '—',
    frozenset(['C2', 'C3', 'C4'])       : '—',
    frozenset(['C1', 'C3', 'C4'])       : '—',
    frozenset(['C1', 'C2', 'C3', 'C4']) : 'S1',
}

UNIQUE = {
    'C1': 'S2, S4, S5',
    'C2': 'S3, S6, S7, S14',
    'C3': '—',
    'C4': '—',
}

COLOURS = {
    'C1': dict(fill='#CECBF6', edge='#534AB7', txt='#26215C'),
    'C2': dict(fill='#9FE1CB', edge='#0F6E56', txt='#04342C'),
    'C3': dict(fill='#F5C4B3', edge='#993C1D', txt='#4A1B0C'),
    'C4': dict(fill='#FAC775', edge='#854F0B', txt='#412402'),
}

OUT_PDF = '../analysis-results/venn_diagram.pdf'
OUT_PNG = '../analysis-results/venn_diagram.png'

# ════════════════════════════════════════════════════════
#  LAYOUT CONSTANTS
# ════════════════════════════════════════════════════════

FIG_W, FIG_H = 19.0, 12.5

CX, CY   = 9.20, 5.80
R        = 2.40
D        = 1.95
ALPHA    = 0.34

# ── Font sizes (all increased) ───────────────────────────
# FS_PANEL_HEADER  = 13.5
# FS_CLUSTER_LABEL = 11.5
# FS_CLUSTER_ID    = 19.0
# FS_GAP_CODE      = 10.5
# FS_GAP_DESC      = 10.5
# FS_STRAT_CODE    = 10.5
# FS_STRAT_DESC    = 10.5
# FS_ANN           = 9.8
# FS_INT           = 10.5
# FS_KEY_HEADER    = 11.0
# FS_KEY_LABEL     = 9.8
# FS_NOTE          = 9.2


# ── Font sizes (substantially larger, same layout) ───────

FS_PANEL_HEADER  = 17
FS_CLUSTER_LABEL = 18
FS_CLUSTER_ID    = 26

FS_GAP_CODE      = 17
FS_GAP_DESC      = 16.5

FS_STRAT_CODE    = 17
FS_STRAT_DESC    = 16.5

FS_ANN           = 16
FS_INT           = 16

FS_KEY_HEADER    = 16
FS_KEY_LABEL     = 16

FS_NOTE          = 10.5

GAP_ROW_H    = 0.42
STRAT_ROW_H  = 0.68
CLUSTER_GAP  = 0.56
KEY_ROW_H    = 0.42

LX, LTY = 0.22, 12.00
RX, RTY = 14.50, 12.00
LW      = 3.80
RW      = 4.10

# ════════════════════════════════════════════════════════
#  DRAWING
# ════════════════════════════════════════════════════════

fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor('white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W);  ax.set_ylim(0, FIG_H)
ax.set_aspect('equal');  ax.axis('off')

CXY = {
    'C1': (CX - D, CY),
    'C2': (CX,     CY + D),
    'C3': (CX + D, CY),
    'C4': (CX,     CY - D),
}

for k, (cx, cy) in CXY.items():
    ax.add_patch(Circle((cx, cy), R,
                         fc=COLOURS[k]['fill'], ec='none',
                         alpha=ALPHA, zorder=2))
    ax.add_patch(Circle((cx, cy), R,
                         fc='none', ec=COLOURS[k]['edge'],
                         lw=1.9, zorder=3))

id_offsets = {
    'C1': (CXY['C1'][0] - 0.98, CXY['C1'][1] + 1.28),
    'C2': (CXY['C2'][0],         CXY['C2'][1] + 1.28),
    'C3': (CXY['C3'][0] + 0.98, CXY['C3'][1] + 1.28),
    'C4': (CXY['C4'][0],         CXY['C4'][1] - 1.28),
}
for k, (x, y) in id_offsets.items():
    ax.text(x, y, k, fontsize=FS_CLUSTER_ID, fontweight='bold',
            color=COLOURS[k]['txt'], ha='center', va='center', zorder=6)

def vtext(x, y, txt, fs, color, bold=False, faint=False):
    ax.text(x, y, txt,
            fontsize   = fs,
            color      = '#AAAAAA' if faint else color,
            fontweight = 'bold' if (bold and not faint) else 'normal',
            style      = 'italic',
            ha='center', va='center',
            alpha      = 0.55 if faint else 1.0,
            zorder     = 8,
            linespacing= 1.5,
            bbox=dict(boxstyle='round,pad=0.18', fc='white',
                      ec='none', alpha=0.55 if faint else 0.62))

cx1,cy1 = CXY['C1']; cx2,cy2 = CXY['C2']
cx3,cy3 = CXY['C3']; cx4,cy4 = CXY['C4']

gap_codes = {k: ', '.join(g[0] for g in v['gaps'])
             for k, v in CLUSTERS.items()}

vtext(cx1 - 0.60, cy1 + 0.16,
      f"Gaps: {gap_codes['C1']}\nUnique: {UNIQUE['C1']}",
      FS_ANN, COLOURS['C1']['txt'])

vtext(cx2, cy2 + 0.72,
      f"Gaps: {gap_codes['C2']}          Unique: {UNIQUE['C2']}",
      FS_ANN, COLOURS['C2']['txt'])

vtext(cx3 + 0.60, cy3 + 0.16,
      f"Gaps: {gap_codes['C3']}\nUnique: {UNIQUE['C3']}",
      FS_ANN, COLOURS['C3']['txt'])

vtext(cx4, cy4 - 0.72,
      f"Gaps: {gap_codes['C4']}          Unique: {UNIQUE['C4']}",
      FS_ANN, COLOURS['C4']['txt'])

int_positions = {
    frozenset(['C1', 'C2'])             : (CX - 1.36, CY + 1.46),
    frozenset(['C2', 'C3'])             : (CX + 1.36, CY + 1.46),
    frozenset(['C1', 'C4'])             : (CX - 1.36, CY - 1.46),
    frozenset(['C3', 'C4'])             : (CX + 1.36, CY - 1.46),
    frozenset(['C1', 'C3'])             : (CX,         CY + 0.60),
    frozenset(['C2', 'C4'])             : (CX,         CY - 0.60),
    frozenset(['C1', 'C2', 'C3'])       : (CX,          CY + 1.42),
    frozenset(['C1', 'C2', 'C4'])       : (CX - 1.42,   CY),
    frozenset(['C2', 'C3', 'C4'])       : (CX + 1.42,   CY),
    frozenset(['C1', 'C3', 'C4'])       : (CX,          CY - 1.42),
}

for region, (x, y) in int_positions.items():
    label = INTERSECTIONS.get(region, '—')
    faint = (label == '—')
    vtext(x, y, label, FS_INT,
          color='#1A1A1A', bold=True, faint=faint)

all4 = frozenset(['C1', 'C2', 'C3', 'C4'])
ax.text(CX, CY, INTERSECTIONS.get(all4, 'S1'),
        fontsize=12.0, fontweight='bold', color='#1A1A1A',
        ha='center', va='center', zorder=10,
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec='#888888', lw=1.3, alpha=0.96))

# ── LEFT PANEL ────────────────────────────────────────────
def panel_header(x, y, txt, width):
    ax.text(x, y, txt, fontsize=FS_PANEL_HEADER, fontweight='bold',
            color='#1A1A1A', ha='left', va='top', zorder=5)
    ax.plot([x, x + width], [y - 0.24, y - 0.24],
            color='#CCCCCC', lw=0.9, zorder=5)

panel_header(LX, LTY, 'Research Gaps', LW)

y_cursor = LTY - 0.44
for k in ['C1', 'C2', 'C3', 'C4']:
    ax.text(LX, y_cursor,
            f"{k} — {CLUSTERS[k]['name']}",
            fontsize=FS_CLUSTER_LABEL, fontweight='bold',
            color=COLOURS[k]['txt'], ha='left', va='top', zorder=5)
    y_cursor -= 0.40

    for code, desc in CLUSTERS[k]['gaps']:
        ax.text(LX + 0.08, y_cursor, code,
                fontsize=FS_GAP_CODE, fontweight='bold',
                color=COLOURS[k]['txt'], ha='left', va='top', zorder=5)
        ax.text(LX + 0.62, y_cursor, desc,
                fontsize=FS_GAP_DESC, color='#2E2E2E',
                ha='left', va='top', zorder=5)
        y_cursor -= GAP_ROW_H

    y_cursor -= CLUSTER_GAP

# ── RIGHT PANEL ───────────────────────────────────────────
panel_header(RX, RTY, 'Mitigation Strategies', RW)

sy = RTY - 0.44
for code, desc in STRATEGIES:
    ax.text(RX + 0.02, sy, code,
            fontsize=FS_STRAT_CODE, fontweight='bold',
            color='#1A1A1A', ha='left', va='top', zorder=5)
    ax.text(RX + 0.60, sy, desc,
            fontsize=FS_STRAT_DESC, color='#2E2E2E',
            ha='left', va='top', zorder=5, linespacing=1.38)
    sy -= STRAT_ROW_H

# ── Colour key ────────────────────────────────────────────
sy -= 0.20
ax.text(RX, sy, 'Cluster colour key:',
        fontsize=FS_KEY_HEADER, fontweight='bold',
        color='#1A1A1A', ha='left', va='top', zorder=5)
sy -= 0.38

key_entries = [
    (k, f"{k} — {CLUSTERS[k]['name']} "
        f"({', '.join(g[0] for g in CLUSTERS[k]['gaps'])})")
    for k in ['C1', 'C2', 'C3', 'C4']
]
for k, lbl in key_entries:
    rect = FancyBboxPatch(
        (RX + 0.02, sy - 0.11), 0.30, 0.22,
        boxstyle='square,pad=0',
        fc=COLOURS[k]['fill'], ec=COLOURS[k]['edge'],
        lw=0.9, alpha=0.85, zorder=5)
    ax.add_patch(rect)
    ax.text(RX + 0.44, sy + 0.01, lbl,
            fontsize=FS_KEY_LABEL, color=COLOURS[k]['txt'],
            ha='left', va='center', zorder=5)
    sy -= KEY_ROW_H

sy -= 0.14
# ax.text(RX, sy,
#         'Codes inside Venn = shared strategies.\n'
#         'Dashes (—) = no strategy documented across those clusters.\n'
#         'Centre (all four clusters): S1 only.',
#         fontsize=FS_NOTE, color='#666666', style='italic',
#         ha='left', va='top', zorder=5, linespacing=1.5)

import os
os.makedirs('analysis-results', exist_ok=True)
fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")