import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv('/data/Deep_Angiography/labeled_DSA_2023_10_24.csv')
counts = df['angio_run'].value_counts()

# ── Code mapping ──────────────────────────────────────────────────────────────
CODE_MAP = {
    'other':                                           'OTH',
    'intrahepatic artery':                             'IHA',
    'external iliac artery, right':                    'EIA-R',
    'celiac trunk':                                    'CT',
    'nondiagnostic':                                   'ND',
    'common hepatic artery':                           'CHA',
    'hepatic artery, right':                           'HA-R',
    'superior mesenteric artery (SMA)':                'SMA',
    'proper hepatic artery':                           'PHA',
    'lower abdominal aorta and aortic bifurcation':    'LAA',
    'hepatic artery, left':                            'HA-L',
    'splenic artery':                                  'SA',
    'internal iliac artery, left':                     'IIA-L',
    'inferior mesenteric artery (IMA)':                'IMA',
    'external iliac artery, left':                     'EIA-L',
    'internal iliac artery, right':                    'IIA-R',
    'renal artery, right':                             'RA-R',
    'unstable':                                        'UNS',
    'upper abdominal aorta':                           'UAA',
    'renal artery, left':                              'RA-L',
    'TRAS, extrenal iliac artery, right':              'TRAS-R',
    'common iliac artery, right':                      'CIA-R',
    'common iliac artery, left':                       'CIA-L',
    'common femoral artery, left':                     'CFA-L',
    'TRAS, extrenal iliac artery, left':               'TRAS-L',
}

coded_counts = counts.rename(index=CODE_MAP)

# ── Figure layout: bar chart (left) + legend table (right) ───────────────────
fig = plt.figure(figsize=(16, 7))
ax_bar  = fig.add_axes([0.05, 0.12, 0.55, 0.80])   # bar chart
ax_leg  = fig.add_axes([0.63, 0.02, 0.36, 0.96])   # legend table
ax_leg.axis('off')

# ── Bar chart ─────────────────────────────────────────────────────────────────
colors = plt.cm.tab20.colors
bars = ax_bar.bar(range(len(coded_counts)), coded_counts.values,
                  color=[colors[i % 20] for i in range(len(coded_counts))],
                  edgecolor='black', linewidth=0.5)

for i, (bar, val) in enumerate(zip(bars, coded_counts.values)):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(val), ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax_bar.set_xticks(range(len(coded_counts)))
ax_bar.set_xticklabels(coded_counts.index, rotation=45, ha='right', fontsize=9)
ax_bar.set_ylabel('Count', fontsize=11)
ax_bar.set_title('angio_run — Count Distribution (coded)', fontsize=12, fontweight='bold')
ax_bar.set_xlim(-0.7, len(coded_counts) - 0.3)
ax_bar.grid(axis='y', linestyle='--', alpha=0.4)

# ── Legend table ──────────────────────────────────────────────────────────────
legend_data = [(code, full, counts[full]) for full, code in CODE_MAP.items()]
legend_data.sort(key=lambda x: -x[2])   # sort by count descending

col_headers = ['Code', 'Full Label', 'N']
table_data  = [[row[0], row[1], row[2]] for row in legend_data]

tbl = ax_leg.table(
    cellText=table_data,
    colLabels=col_headers,
    loc='center',
    cellLoc='left',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.18)

# Style header row
for col in range(3):
    tbl[(0, col)].set_facecolor('#2c3e50')
    tbl[(0, col)].set_text_props(color='white', fontweight='bold')

# Alternate row shading + color-code the Code cell
for row_idx, (code, full, n) in enumerate(legend_data, start=1):
    bg = '#f0f4f8' if row_idx % 2 == 0 else 'white'
    for col in range(3):
        tbl[(row_idx, col)].set_facecolor(bg)
    # match bar color
    orig_idx = list(CODE_MAP.values()).index(code)
    tbl[(row_idx, 0)].set_facecolor(colors[orig_idx % 20])
    tbl[(row_idx, 0)].set_text_props(fontweight='bold')

tbl.auto_set_column_width([0, 1, 2])
ax_leg.set_title('Code Reference', fontsize=10, fontweight='bold', pad=4)

plt.suptitle('angio_run Category Distribution', fontsize=14, fontweight='bold', y=1.01)
plt.savefig('angio_run_coded.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → angio_run_coded.png")