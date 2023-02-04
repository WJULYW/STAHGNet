import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
mat = np.array([[0, 0.0652, -0.0726], [0.142, 0, 0.0647], [0.0051, -0.0929, 0]])
sns.set_theme()

ax = sns.heatmap(mat, cmap="GnBu", vmax=0.25, vmin=-0.35)
ax.set_xticklabels([180, 183, 324])
ax.set_yticklabels([180, 183, 324])
plt.show()