from matplotlib import pyplot as plt
import pandas as pd

names = ['3s5z.csv', '3s5z_vs_3s6z.csv', '10m_vs_11m.csv', '27m_vs_30m.csv', '8m_vs_9m.csv']
easy = ['3s5z.csv', '10m_vs_11m.csv']
hard = ['8m_vs_9m.csv']
super_hard = ['27m_vs_30m.csv', '3s5z_vs_3s6z.csv']

fig, axes = plt.subplots(5, 1, figsize=(4, 10))
axes = axes.flatten()

axvline_steps = [1050000, 950000, 400000, 1600000, None]

for i, n in enumerate(easy+hard+super_hard):
    df_origin = pd.read_csv(n)
    df_random = pd.read_csv(n[:-4] + '_random' + '.csv')
    
    axes[i].plot(df_origin['Step'], df_origin['Value'], color=f'C{i}', linestyle=':', label=f'{n[:-4]} origin', linewidth=2)
    axes[i].plot(df_random['Step'], df_random['Value'], color=f'C{i}', label=f'{n[:-4]} random', linewidth=2)
    if axvline_steps[i]:
        axes[i].axvline(x=axvline_steps[i], color=f'black')

    axes[i].set_xlim(-100000, 300_0000)
    axes[i].set_ylim(-0.05, 1.03)
    axes[i].legend()
    axes[i].grid(True)

# axes[-1].axis('off')
plt.tight_layout()
plt.show()
