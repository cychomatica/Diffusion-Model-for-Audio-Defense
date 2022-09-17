import numpy as np
import matplotlib.pyplot as plt

# eot_size=1
# attack_steps = [0, 50, 100, 150, 200, 250, 300]
# acc_vanilla = [100, 0, 0, 0, 0, 0, 0]
# acc_advtr = [100, 23, 21, 18, 18, 18, 18]
# acc_diffusion = [96, 77, 72, 73, 66, 63, 66]

# plt.figure(dpi=1000)
# plt.title('EOT_size=1')
# plt.xlabel('PGD steps')
# plt.ylabel('accuracy')
# plt.ylim(0,100)
# plt.plot(attack_steps, acc_vanilla, '-o', label='vanilla')
# plt.plot(attack_steps, acc_advtr, '-s', label='adversarial training')
# plt.plot(attack_steps, acc_diffusion, '-*', label='diffusion purifucation')
# plt.legend()
# plt.savefig('_Experiments/curves/PGD&acc_EOT=1.png')

# PGD_steps=10
# eot_size = [1, 50, 100, 150, 200]
# acc_vanilla = [6, 3, 3, 6, 3]
# acc_advtr = [45, 45, 45, 45, 45]
# acc_diffusion = [84, 82, 81, 80, 80]

# plt.figure(dpi=1000)
# plt.title('PGD_steps=10')
# plt.xlabel('EOT size')
# plt.ylabel('accuracy')
# plt.ylim(0,100)
# plt.plot(eot_size, acc_vanilla, '-o', label='vanilla')
# plt.plot(eot_size, acc_advtr, '-s', label='adversarial training')
# plt.plot(eot_size, acc_diffusion, '-*', label='diffusion purifucation')
# plt.legend(bbox_to_anchor=(0.5, 0.3))
# plt.savefig('_Experiments/curves/EOT&acc_PGD=10.png')

# PGD_steps=10
diffusion_steps = [0, 1, 2, 3, 5, 7, 10]
acc_eps_65 = [6, 94, 90, 89, 84, 77, 67]
acc_eps_131 = [0, 76, 89, 86, 83, 74, 66]
acc_eps_262 = [0, 27, 70, 85, 84, 74, 68]
acc_eps_524 = [0, 0, 21, 53, 69, 57]

plt.figure(dpi=1000)
plt.title('')
plt.xlabel('diffusion steps')
plt.ylabel('accuracy')
plt.xlim(0,10)
plt.ylim(0,100)
plt.plot(diffusion_steps[:len(acc_eps_65)], acc_eps_65, '-o', label='eps=0.002')
plt.plot(diffusion_steps[:len(acc_eps_131)], acc_eps_131, '-s', label='eps=0.004')
plt.plot(diffusion_steps[:len(acc_eps_262)], acc_eps_262, '-*', label='eps=0.008')
plt.plot(diffusion_steps[:len(acc_eps_524)], acc_eps_524, '-^', label='eps=0.016')
plt.legend()
# plt.legend(bbox_to_anchor=(0.5, 0.3))
plt.savefig('_Experiments/curves/diffusion_steps&eps&acc_PGD=10.png')