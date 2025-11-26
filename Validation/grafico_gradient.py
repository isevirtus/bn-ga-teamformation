import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize":10,
})

df = pd.read_csv("gradient_check_results.csv")

sub_at = df[df["kind"] == "AT|ACfixed"].copy()
sub_ac = df[df["kind"] == "AC|ATfixed"].copy()

linestyles = ['-', '--', '-.']
markers = ['o', 's', 'D']

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# -------- (a) AE vs AT (fixed AC) --------
ax = axes[0]
for i, ac in enumerate(sorted(sub_at["ac_fixed"].unique())):
    ss = sub_at[sub_at["ac_fixed"] == ac]
    ax.plot(
        ss["at_base"],
        ss["AE(base)"],
        marker=markers[i % len(markers)],
        linestyle=linestyles[i % len(linestyles)],
        color='black',
        label=f"{ac:.1f}",
    )

ax.set_xlabel("AT")
ax.set_ylabel(r"$\mathbb{E}[\mathrm{AE}]$")
ax.set_title("(a) AE vs AT (fixed AC)")
ax.grid(True, linestyle=':', linewidth=0.5)

# legenda abaixo do subplot (a)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),  # x centralizado, y abaixo do eixo
    ncol=3,
    frameon=False
)

# -------- (b) AE vs AC (fixed AT) --------
ax = axes[1]
for i, at in enumerate(sorted(sub_ac["at_fixed"].unique())):
    ss = sub_ac[sub_ac["at_fixed"] == at]
    ax.plot(
        ss["ac_base"],
        ss["AE(base)"],
        marker=markers[i % len(markers)],
        linestyle=linestyles[i % len(linestyles)],
        color='black',
        label=f"{at:.1f}",
    )

ax.set_xlabel("AC")
ax.set_ylabel(r"")
ax.set_title("(b) AE vs AC (fixed AT)")
ax.grid(True, linestyle=':', linewidth=0.9)

# legenda abaixo do subplot (b)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    frameon=False
)

# dá espaço extra embaixo para caber as legendas
plt.subplots_adjust(bottom=0.3)

plt.savefig("fig_sensitivity_local.png", dpi=300, bbox_inches="tight")
plt.show()
