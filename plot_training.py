# Plot gp training
import numpy as np
import matplotlib.pyplot as plt
# import torch
# import seaborn as sns
# sns.set_theme()

def getMeanAndStd(input):
    _num_convolute = 10
    _mean = np.zeros(shape=[len(input)])
    _std = np.zeros(shape=[len(input)])
    for i in range(len(input)):
        _mean[i] = np.mean(input[0+np.abs(i-_num_convolute//2):i+1+_num_convolute//2])
        _std[i] = np.std(input[0+np.abs(i-_num_convolute//2):i+1+_num_convolute//2])
    return _mean, _std

def plot(data, label:str=''):
    _mean, _std = getMeanAndStd(data)
    plt.plot(np.arange(len(data)), _mean, label=label)
    plt.fill_between(np.arange(len(data)), _mean+_std, _mean-_std, alpha=0.05)


# _wk_reward_5 = np.load('/home/amornyos/PhD/Packages/resources/test_results/GPDQ_EXACTRBF/walker2d-medium-expert-v2_GPDQ_EXACTRBF_test_results.npz')
# _wk_reward_5 = _wk_reward_5['arr_0']

# _wk_reward_6 = np.load('/home/amornyos/PhD/Packages/resources/test_results/GPDQ_EXACTNZMRBF2/walker2d-medium-v2_GPDQ_EXACTNZMRBF2_test_results.npz')
# _wk_reward_6 = _wk_reward_6['arr_0']

# _wk_reward_7 = np.load('/home/amornyos/PhD/Packages/resources/test_results/GPDQ_EXACTMATERN/walker2d-medium-expert-v2_GPDQ_EXACTMATERN_test_results.npz')
# _wk_reward_7 = _wk_reward_7['arr_0']

# _wk_reward_8 = np.load('/home/amornyos/PhD/Packages/resources/test_results/GPDQ_DOUBLEQEXACTRBF/walker2d-medium-expert-v2_GPDQ_DOUBLEQEXACTRBF_test_results.npz')
# _wk_reward_8 = _wk_reward_8['arr_0']

# _wk_reward_9 = np.load('/home/amornyos/PhD/Packages/resources/test_results/GPDP_EXACTMATERN/walker2d-medium-expert-v2_GPDP_EXACTMATERN_test_results.npz')
# _wk_reward_9 = _wk_reward_9['arr_0']

_wk_reward_10 = np.load('/home/amornyos/PhD/Packages/resources/test_results/GPDQ_EXACTRBF/hopper-medium-v2_GPDQ_EXACTRBF_test_results_coeff_10_gpsize_1000.npz')
_wk_reward_10 = _wk_reward_10['arr_0']

_wk_reward_11 = np.load('/home/amornyos/PhD/Packages/resources/test_results/GPDQ_EXACTRBF/hopper-medium-v2_GPDQ_EXACTRBF_test_results_coeff_3_gpsize_1000.npz')
_wk_reward_11 = _wk_reward_11['arr_0']

_wk_reward_12 = np.load('/home/amornyos/PhD/Packages/resources/test_results/GPDQ_EXACTRBF/hopper-medium-v2_GPDQ_EXACTRBF_test_results.npz')
_wk_reward_12 = _wk_reward_12['arr_0']

# plot(_wk_reward_5, label='2000+128+clipped')
# plot(_wk_reward_6, label='1000+128+standard')
# plot(_wk_reward_7, label='1000+8+standard')
# plot(_wk_reward_8, label='normal+rbf')
# plot(_wk_reward_9, label='normal+rbf+250')
plot(_wk_reward_10, label='coeff_1_gpsize_1000')
plot(_wk_reward_11, label='coeff_3_gpsize_1000')
plot(_wk_reward_12, label='coeff_10_gpsize_1000')


# print('Max(exact): ', np.max(_wk_reward_5))
# print('Max(exact): ', np.max(_wk_reward_6))
# print('Max(exact): ', np.max(_wk_reward_7))
# print('Max(exact): ', np.max(_wk_reward_8))
# print('Max(exact): ', np.max(_wk_reward_9))
print('Max(exact): ', np.max(_wk_reward_10))
print('Max(exact): ', np.max(_wk_reward_11))
print('Max(exact): ', np.max(_wk_reward_12))


plt.ylim(0, 120)
plt.legend()
plt.show()