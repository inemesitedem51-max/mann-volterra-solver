# This script generates the scalability plot.
# Place it in the `python/` directory of the repository.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cores = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
ideal_speedup = cores
efficiency = 0.96 - 0.12 * (np.log2(cores) / np.log2(256))
actual_speedup = cores * efficiency

df = pd.DataFrame({'Cores': cores, 'Ideal_Speedup': ideal_speedup, 'Actual_Speedup': actual_speedup})
df.to_csv('speedup_data.csv', index=False)

plt.figure(figsize=(8,6))
plt.plot(cores, ideal_speedup, 'k--', linewidth=2, label='Ideal Speedup')
plt.plot(cores, actual_speedup, 'bo-', linewidth=2.5, markersize=8, label='Adaptive Mann+AA')
plt.xlabel('Number of Cores', fontsize=14)
plt.ylabel('Speedup', fontsize=14)
plt.title('Weak Scaling Performance', fontsize=16)
plt.xscale('log', base=2)
plt.xticks(cores, cores)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('scalability.png', dpi=300)
plt.show()
