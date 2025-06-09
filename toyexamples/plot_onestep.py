import pickle 
import numpy as np 
import matplotlib.pyplot as plt

with open('results/onestep.pkl', 'rb') as file:
    results = pickle.load(file)

iters = range(len(results['admm_losses']) // 2) 

fig, ax = plt.subplots(figsize=(6, 3))

plt.axhline(y=161.1069, color='gray', linestyle='--', linewidth=4)
plt.plot(iters, results['admm_losses'][::2], linewidth=2, color="gray", marker="d", markersize=10, markerfacecolor='white', markeredgecolor="gray", label="FederatedADMM")
plt.plot(iters, results['bregman_admm_losses'][::2], linewidth=2, color="lightgray", marker="s", markersize=10, markerfacecolor='white', markeredgecolor="lightgray", label="BregmanADMM")
plt.plot(iters, results['bayes_admm_losses'][::2], linewidth=2, color="k", marker="o", markersize=10, markerfacecolor='white', markeredgecolor="k", label="BayesADMM")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

# Set axis markers and labels to gray
plt.tick_params(axis='both', colors='gray')  # Axis ticks
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.ylim([155, 260])
plt.xlabel('Communication Round', color='gray', fontsize=14)
plt.ylabel('Global Loss at Client 1', color='gray', fontsize=14)
plt.legend(fontsize=14,frameon=False)

#plt.show()
plt.tight_layout()

plt.savefig('results/onestep.pdf', format='pdf')
