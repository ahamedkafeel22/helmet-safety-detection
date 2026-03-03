import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── LOAD CSV ──────────────────────────────────────────
df = pd.read_csv(r"C:\Users\syedk\Documents\Self Projects\Project 4\safety_report.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🦺 Helmet Safety Compliance Dashboard', fontsize=16, fontweight='bold')

# ── PLOT 1: Compliance Rate Gauge ─────────────────────
ax1 = axes[0, 0]
compliance = 72.49
colors = ['#2ecc71' if compliance >= 80 else '#e74c3c', '#ecf0f1']
wedges, _ = ax1.pie([compliance, 100-compliance],
                     colors=colors, startangle=90,
                     wedgeprops=dict(width=0.4))
ax1.text(0, 0, f'{compliance:.1f}%', ha='center', va='center',
         fontsize=24, fontweight='bold',
         color='#2ecc71' if compliance >= 80 else '#e74c3c')
ax1.set_title('Overall Compliance Rate', fontweight='bold')

# ── PLOT 2: Helmet vs Violation ────────────────────────
ax2 = axes[0, 1]
categories = ['Helmets\nDetected', 'Violations\n(No Helmet)']
values = [df['helmet_count'].sum(), df['head_count'].sum()]
colors2 = ['#2ecc71', '#e74c3c']
bars = ax2.bar(categories, values, color=colors2, width=0.5, edgecolor='white')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha='center', fontweight='bold', fontsize=12)
ax2.set_title('Helmet vs Violation Count', fontweight='bold')
ax2.set_ylabel('Count')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── PLOT 3: Compliance Distribution ───────────────────
ax3 = axes[1, 0]
ax3.hist(df['compliance_%'], bins=20, color='#3498db',
         edgecolor='white', alpha=0.8)
ax3.axvline(x=100, color='#2ecc71', linestyle='--', linewidth=2, label='100% Safe')
ax3.axvline(x=df['compliance_%'].mean(), color='#e74c3c',
            linestyle='--', linewidth=2, label=f'Mean: {df["compliance_%"].mean():.1f}%')
ax3.set_title('Compliance Rate Distribution', fontweight='bold')
ax3.set_xlabel('Compliance %')
ax3.set_ylabel('Number of Images')
ax3.legend()
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── PLOT 4: Safe vs Violation Images ──────────────────
ax4 = axes[1, 1]
status_counts = df['status'].value_counts()
colors4 = ['#2ecc71' if s == 'SAFE' else '#e74c3c' for s in status_counts.index]
wedges, texts, autotexts = ax4.pie(status_counts.values,
                                    labels=status_counts.index,
                                    colors=colors4,
                                    autopct='%1.1f%%',
                                    startangle=90)
ax4.set_title('Safe vs Violation Images', fontweight='bold')

plt.tight_layout()
plt.savefig(r"C:\Users\syedk\Documents\Self Projects\Project 4\safety_dashboard.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Dashboard saved!")