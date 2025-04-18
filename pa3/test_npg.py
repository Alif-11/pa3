import pickle
import numpy as np
import npg_utils


with open('test_info.pkl', 'rb') as f:
    tests_info = pickle.load(f)
test_cases = sorted(tests_info.keys())

""" ------------- testing action distribution computation ----------------"""
print('-'*10 + ' testing compute_action_distribution ' + '-'*10)
for i in test_cases:
    theta = tests_info[i]['theta']
    phis = tests_info[i]['phis']
    soln_action_dist = tests_info[i]['action_dst']
    action_dist = npg_utils.compute_action_distribution(theta, phis)
    assert soln_action_dist.shape == action_dist.shape, 'compute_action_dist output shape incorrect'

    err = np.linalg.norm(soln_action_dist - action_dist)
    print('test {} for compute_action_distribution - error = {}'.format(i, err))

""" ------------- testing compute_log_softmax_grad ----------------"""
print('-' * 10 + ' testing compute_log_softmax_grad ' + '-' * 10)
for i in test_cases:
    theta = tests_info[i]['theta']
    phis = tests_info[i]['phis']
    action = tests_info[i]['action']
    soln_grad = tests_info[i]['grad']
    grad = npg_utils.compute_log_softmax_grad(theta, phis, action)
    assert soln_grad.shape == grad.shape, "compute_log_softmax_grad output shape incorrect"
    err = np.linalg.norm(soln_grad - grad)
    print('test {} for compute_log_softmax_grad - error = {}'.format(i, err))


""" ------------- testing compute_fisher_matrix ----------------"""
print('-' * 10 + ' testing compute_fisher_matrix ' + '-' * 10)
for i in test_cases:
    total_grads = tests_info[i]['total_grads']
    total_rewards = tests_info[i]['total_rewards']

    soln_fisher = tests_info[i]['fisher']
    fisher = npg_utils.compute_fisher_matrix(total_grads)
    assert soln_fisher.shape == fisher.shape, 'compute_fisher_matrix output shape incorrect'
    err = np.linalg.norm(soln_fisher - fisher)
    print('test {} for compute_fisher_matrix - error = {}'.format(i, err))

""" ------------- testing compute_value_gradient ----------------"""
print('-' * 10 + ' testing compute_value_gradient ' + '-' * 10)
for i in test_cases:
    total_grads = tests_info[i]['total_grads']
    total_rewards = tests_info[i]['total_rewards']

    soln_v_grad = tests_info[i]['v_grad']
    v_grad = npg_utils.compute_value_gradient(total_grads, total_rewards)
    assert soln_v_grad.shape == v_grad.shape, "compute_value_gradient output shape incorrect"
    err = np.linalg.norm(soln_v_grad - v_grad)
    print('test {} for compute_value_gradient - error = {}'.format(i, err))

""" ------------- testing compute_eta ----------------"""
print('-' * 10 + ' testing compute_eta ' + '-' * 10)
for i in test_cases:

    fisher = tests_info[i]['fisher']
    delta = 1e-2
    v_grad = tests_info[i]['v_grad']
    soln_eta = tests_info[i]['eta']

    eta = npg_utils.compute_eta(delta, fisher, v_grad)

    err = np.linalg.norm(soln_eta - eta)
    print('test {} for compute_eta - error = {}'.format(i, err))

