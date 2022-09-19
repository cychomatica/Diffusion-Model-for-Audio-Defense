import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, interp1d
from scipy.optimize import curve_fit

# region
eot_size=1
attack_steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
acc_vanilla = [100, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
acc_advtr = [100, 45, 28, 24, 23, 23, 22, 23, 20, 21, 22, 19]
acc_diffusion = [96, 84, 77, 77, 76, 77, 74, 72, 74, 70, 72, 73]

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

x = np.array(attack_steps)
y1, y2, y3 = np.array(acc_vanilla), np.array(acc_advtr), np.array(acc_diffusion)
popt1, pcov1 = curve_fit(func, x, y1, maxfev=74751)
popt2, pcov2 = curve_fit(func, x, y2, maxfev=74751)
popt3, pcov3 = curve_fit(func, x, y3, bounds=([-np.inf, -10, -np.inf], [np.inf, 10, np.inf]), maxfev=74751)   

x_new = np.linspace(min(attack_steps), max(attack_steps), 1000)
# y1_new = smooth_acc_vanilla(x_new)
# y2_new = smooth_acc_advtr(x_new)
# y3_new = smooth_acc_diffusion(x_new)

plt.figure(dpi=1000)
plt.title('')
plt.xlabel('PGD steps')
plt.ylabel('accuracy')
plt.ylim(0, 100)
plt.plot(x_new, func(x_new, *popt1), color='darkorange', label='vanilla')
plt.plot(x_new, func(x_new, *popt2), color='steelblue', label='adversarial training')
plt.plot(x_new, func(x_new, *popt3), color='forestgreen', label='diffusion purifucation')
plt.plot(attack_steps, acc_vanilla, 'o', color='darkorange')
plt.plot(attack_steps, acc_advtr, 's', color='steelblue')
plt.plot(attack_steps, acc_diffusion, '*', color='forestgreen')
# plt.plot(attack_steps, acc_vanilla, '-o', label='vanilla')
# plt.plot(attack_steps, acc_advtr, '-s', label='adversarial training')
# plt.plot(attack_steps, acc_diffusion, '-*', label='diffusion purifucation')
plt.legend()
plt.savefig('_Experiments/curves/prettified_PGD&acc.png')
# endregion

# region
# PGD_steps=70
# eot_size = [1, 5, 10, 15, 20, 25]
# acc_vanilla = [1, 1, 1, 1, 1, 1]
# acc_advtr = [23, 22, 22, 21, 22, 22]
# acc_diffusion = [72, 75, 74, 73, 74, 73]

# plt.figure(dpi=1000)
# plt.title('')
# plt.xlabel('EOT size')
# plt.ylabel('accuracy')
# plt.ylim(0,100)
# plt.plot(eot_size[:len(acc_vanilla)], acc_vanilla, '-o', label='vanilla')
# plt.plot(eot_size[:len(acc_advtr)], acc_advtr, '-s', label='adversarial training')
# plt.plot(eot_size[:len(acc_diffusion)], acc_diffusion, '-*', label='diffusion purifucation')
# # plt.legend(bbox_to_anchor=(0.5, 0.3))
# plt.legend()
# plt.savefig('_Experiments/curves/EOT&acc_PGD={}.png'.format(PGD_steps))
# endregion

# region
# PGD_steps=10
# diffusion_steps = [0, 1, 2, 3, 5, 7, 10]
# acc_eps_65 = [6, 94, 90, 89, 84, 77, 67]
# acc_eps_131 = [0, 76, 89, 86, 83, 74, 66]
# acc_eps_262 = [0, 27, 70, 85, 84, 74, 68]
# acc_eps_524 = [0, 0, 21, 53, 69, 57, 63]

# plt.figure(dpi=1000)
# plt.title('')
# plt.xlabel('diffusion steps')
# plt.ylabel('accuracy')
# plt.xlim(0,10)
# plt.ylim(0,100)
# plt.plot(diffusion_steps[:len(acc_eps_65)], acc_eps_65, '-o', label='eps=0.002')
# plt.plot(diffusion_steps[:len(acc_eps_131)], acc_eps_131, '-s', label='eps=0.004')
# plt.plot(diffusion_steps[:len(acc_eps_262)], acc_eps_262, '-*', label='eps=0.008')
# plt.plot(diffusion_steps[:len(acc_eps_524)], acc_eps_524, '-^', label='eps=0.016')
# plt.legend()
# # plt.legend(bbox_to_anchor=(0.5, 0.3))
# plt.savefig('_Experiments/curves/diffusion_steps&eps&acc_PGD={}.png'.format(PGD_steps))
# endregion