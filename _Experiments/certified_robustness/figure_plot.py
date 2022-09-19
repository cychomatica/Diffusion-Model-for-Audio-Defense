import numpy as np
import matplotlib.pyplot as plt
import json, os

N = 100000
sigma = '1.0'

radius_array = np.arange(0,4.05,0.05)
# diffusion = np.load('records/num_examples=100_N={}_sigma={}_defense=diffusion.npy'.format(N,sigma))
# diffusion_2 = np.load('records/2ShotsRev/num_examples=100_N={}_sigma={}_defense=diffusion.npy'.format(N,sigma))
# diffusion_path = '_Experiments/certified_robustness/records/diffusion_N=100000/{}'.format(sigma)
diffusion_path = '_Experiments/certified_robustness/records/diffusion_N=100000/1.00'
diffusion_data = []
diffusion_certified_accuracy = np.zeros_like(radius_array, dtype=float)
files_list = os.listdir(diffusion_path)
files_list.sort(key=lambda x: int(x[:-5]))
for item in files_list:
    file_path = os.path.join(diffusion_path, item)
    with open(file_path, 'r') as f:
        data = json.load(f)
        diffusion_data.append(data)
        f.close()

diffusion_total = len(diffusion_data)
for i in range(len(radius_array)):
    diffusion_certified_correct = 0
    for j in range(diffusion_total):
        if diffusion_data[j]['y_pred'] == diffusion_data[j]['y_true'] \
            and diffusion_data[j]['certified_radius'] >= radius_array[i]:
            diffusion_certified_correct = diffusion_certified_correct + 1
    diffusion_certified_accuracy[i] = 100 * diffusion_certified_correct / diffusion_total

randsmooth_gaussian = np.load('_Experiments/certified_robustness/records/randsmooth_gaussian_aug_N=100000/num_examples=100_N={}_sigma={}_defense=randsmooth.npy'.format(N,sigma))
randsmooth_vanilla = np.load('_Experiments/certified_robustness/records/randsmooth_N=100000/num_examples=100_N={}_sigma={}_defense=randsmooth.npy'.format(N,sigma))
# randsmooth_certified_accuracy = np.load('_Experiments/certified_robustness/records/randsmooth_N=100000/num_examples=100_N={}_sigma={}_defense=randsmooth.npy'.format(N,sigma))

plt.figure(dpi=1000)
plt.title('sigma={}, N={}'.format(sigma,N))
plt.xlabel('radius')
plt.ylabel('certified accuracy')
plt.plot(radius_array, diffusion_certified_accuracy, label='diffusion smoothing')
# plt.plot(radius_array, diffusion_2, label='diffusion smoothing (two-shots-rev)')
plt.plot(radius_array, randsmooth_gaussian, label='randomized smoothing (gaussian)')
plt.plot(radius_array, randsmooth_vanilla, label='randomized smoothing (vanilla)')
plt.legend()
plt.savefig('_Experiments/certified_robustness/figures/_num_examples=100_N={}_sigma={}.png'.format(N,sigma))

# diffusion_path = '_Experiments/certified_robustness/records/diffusion_N=100000/{}'.format(sigma)
# diffusion_data = []
# diffusion_certified_accuracy = np.zeros_like(radius_array, dtype=float)
# files_list = os.listdir(diffusion_path)
# files_list.sort(key=lambda x: int(x[:-5]))
# for item in files_list:
#     file_path = os.path.join(diffusion_path, item)
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#         diffusion_data.append(data)
#         f.close()

# diffusion_total = len(diffusion_data)
# for i in range(len(radius_array)):
#     diffusion_certified_correct = 0
#     for j in range(diffusion_total):
#         if diffusion_data[j]['y_pred'] == diffusion_data[j]['y_true'] \
#             and diffusion_data[j]['certified_radius'] >= radius_array[i]:
#             diffusion_certified_correct = diffusion_certified_correct + 1
#     diffusion_certified_accuracy[i] = 100 * diffusion_certified_correct / diffusion_total


# randsmooth_path = '_Experiments/certified_robustness/records/randsmooth_gaussian_aug_N={}/sigma={}/sigma={}_N={}.json'.format(N, sigma, sigma, N)
# randsmooth_data = None
# randsmooth_certified_accuracy = np.zeros_like(radius_array, dtype=float)
# with open(randsmooth_path, 'r') as f:
#     randsmooth_data = json.load(f)
#     f.close()

# randsmooth_total = len(randsmooth_data)
# for i in range(len(radius_array)):
#     randsmooth_certified_correct = 0
#     for j in range(randsmooth_total):
#         if randsmooth_data[j]['y_pred'] == randsmooth_data[j]['y_true'] \
#             and randsmooth_data[j]['certified_radius'] >= radius_array[i]:
#             randsmooth_certified_correct = randsmooth_certified_correct + 1
#     randsmooth_certified_accuracy[i] = 100 * randsmooth_certified_correct / randsmooth_total

# plt.figure(dpi=1000)
# plt.title('sigma={}, N={}'.format(sigma,N))
# plt.xlabel('radius')
# plt.ylabel('certified accuracy')
# plt.plot(radius_array, diffusion_certified_accuracy, label='diffusion smoothing')
# plt.plot(radius_array, randsmooth_certified_accuracy, label='randomized smoothing')
# plt.legend()
# plt.savefig('_Experiments/certified_robustness/figures/num_examples=100_N={}_sigma={}.png'.format(N,sigma))