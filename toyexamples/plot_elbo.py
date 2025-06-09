import pickle 
import numpy as np 
import matplotlib.pyplot as plt

with open('results/neg-elbo.pkl', 'rb') as file:
    results = pickle.load(file)

niters = len(results['bregman_admm_losses']) - 1
iters = range(niters // 2) 

#print(len(iters),len(results['bregman_admm_losses'][0:niters:2]))

fig, ax = plt.subplots(figsize=(6, 3))

plt.axhline(y=2312, color='gray', linestyle='--', linewidth=4)
plt.plot(iters, results['pvi_losses'][0:niters:2], linewidth=2, color="red", marker="d", markersize=10, markerfacecolor='white', markeredgecolor="red", label="PVI (no damping)")
plt.plot(iters, results['pvi_damped_losses'][0:niters:2], linewidth=2, color="gray", marker="d", markersize=10, markerfacecolor='white', markeredgecolor="gray", label="PVI (with damping)")
plt.plot(iters, results['bregman_admm_losses'][0:niters:2], linewidth=2, color="lightgray", marker="s", markersize=10, markerfacecolor='white', markeredgecolor="lightgray", label="BregmanADMM")
plt.plot(iters, results['bayes_admm_losses'][0:niters:2], linewidth=2, color="k", marker="o", markersize=10, markerfacecolor='white', markeredgecolor="k", label="BayesADMM")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

# Set axis markers and labels to gray
plt.tick_params(axis='both', colors='gray')  # Axis ticks
plt.xticks([0, 2, 4, 6, 8, 10, 12])
plt.ylim([6000, 20000])
plt.xlabel('Communication Round', color='gray', fontsize=14)
plt.ylabel('Variational Objective', color='gray', fontsize=14)
plt.legend(fontsize=14,frameon=False)

#plt.show()
plt.tight_layout()

plt.savefig('results/elbo.pdf', format='pdf')
