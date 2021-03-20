import os
import itertools
import pdb
import argparse
import time

def write_sbatch_conf(f, exp_name = "default", grace_partition = "gpu", gpu_type = "any", dir_name = "./"):
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name=' + exp_name + '\n')
    f.write('#SBATCH --ntasks=1 --nodes=1\n')
    f.write('#SBATCH --partition=' + grace_partition + '\n')
    f.write('#SBATCH --mem=8G\n')
    f.write('#SBATCH --cpus-per-task=8\n')
    f.write('#SBATCH --time=48:00:00\n')
    if gpu_type != "any":
        f.write('#SBATCH --gres=gpu:' + str(gpu_type) + ':1\n')
    else:
        f.write('#SBATCH --gres=gpu:1\n')
    f.write('#SBATCH --output=' + dir_name + exp_name + '/log\n')
    f.write('module load miniconda\n')
    f.write('conda activate py37_dev\n')

def generate_cmd(exp_name = "default", dataset = "CIFAR10", num_classes = 10, model = "VGG5", iid = True, optimizer = "Adam", bs = 64, lr = 0.0001, lr_reduce = 5, epochs = 10, local_ep = 5, eval_every = 1, num_users = 20, frac = 0.1, gpu = 0, timesteps = 20, dir_name = "./", ann_baseline = False, straggler_prob = 0.0, no_batchnorm = False, grad_noise_stdev = 0.0):
    s = 'python main_fed.py '
    s = s + '--dataset ' + dataset + ' '
    s = s + '--num_classes ' + str(num_classes) + ' '
    s = s + '--model ' + model + ' '
    if iid == True:
        s = s + '--iid '
    if no_batchnorm == True:
        s = s + '--no_batchnorm '
    s = s + '--optimizer '+ optimizer + ' '
    s = s + '--bs ' + str(bs) + ' '
    s = s + '--local_bs ' + str(bs) + ' '
    s = s + '--lr ' + str(lr) + ' '
    s = s + '--lr_reduce ' + str(lr_reduce) + ' '
    s = s + '--epochs ' + str(epochs) + ' '
    s = s + '--local_ep ' + str(local_ep) + ' '
    s = s + '--eval_every ' + str(eval_every) + ' '
    s = s + '--num_users ' + str(num_users) + ' '
    s = s + '--frac ' + str(frac) + ' '
    s = s + '--gpu ' + str(gpu) + ' '
    s = s + '--timesteps ' + str(timesteps) + ' '
    if ann_baseline == False:
        s = s + '--snn '
    s = s + '--straggler_prob ' + str(straggler_prob) + ' '
    s = s + '--grad_noise_stdev ' + str(grad_noise_stdev) + ' '
    s = s + '--result_dir ' + dir_name + exp_name + '/snn_bntt '
    s = s + '\n'
    return s

def generate_script(file_name, gpu_type = "any", grace_partition = "gpu", exp_name = "default", dataset = "CIFAR10", num_classes = 10, model = "VGG5", iid = True, optimizer = "Adam", bs = 64, lr = 0.0001, lr_reduce = 5, epochs = 10, local_ep = 5, eval_every = 1, num_users = 20, frac = 0.1, gpu = 0, timesteps = 20, dir_name = "./", ann_baseline = False, straggler_prob = 0.0, no_batchnorm = False, grad_noise_stdev = 0.0):
    f = open(file_name, 'w', buffering = 1)
    write_sbatch_conf(f, dir_name = dir_name, exp_name = exp_name, grace_partition = grace_partition, gpu_type = gpu_type)
    s = generate_cmd(exp_name = exp_name, dataset = dataset, num_classes = num_classes, model = model, iid = iid, optimizer = optimizer, bs = bs, lr = lr, lr_reduce = lr_reduce, epochs = epochs, local_ep = local_ep, eval_every = eval_every, num_users = num_users, frac = frac, gpu = gpu, timesteps = timesteps, dir_name = dir_name, ann_baseline = ann_baseline, straggler_prob = straggler_prob, no_batchnorm = no_batchnorm, grad_noise_stdev = grad_noise_stdev)
    f.write(s)
    f.close()

def main():
    grace_partition = "gpu"
    gpu_type = "any"
    dataset = "CIFAR100"
    num_classes = 100
    model = "VGG9"
    bs = 32
    eval_every = 1
    optimizer = "SGD"
    lr = 0.1
    lr_reduce = 5
    exp_name = "fed_snn_bntt_cifar100"
    epochs = 100
    local_ep = 2
    gpu = 0
    timesteps = 20
    # straggler_prob_list = [0.0, 0.25, 0.5, 0.75, 0.95]
    straggler_prob_list = [0.0]
    ann_baseline = False
    no_batchnorm = False
    # grad_noise_stdev_list = [0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    grad_noise_stdev_list = [0.0]
    
    user_combs = [[1, 1], [20, 0.25], [10, 0.2], [5, 1], [100, 0.1], [150, 0.1], [200, 0.1]]
    # user_combs = [[100, 0.1]]
    for rep in range(1):
        dir_name = "./bntt_experiments_extended_cifar100_vgg9_run4_repeat"+str(rep)+"/"
        for straggler_prob in straggler_prob_list:
            for grad_noise_stdev in grad_noise_stdev_list:
                for comb in user_combs:
                    iid = False
                    num_users = comb[0]
                    frac = comb[1]
                    exp_name = "fed_snn_bntt_nc_" + str(num_users) + "_frac_" + str(frac) + "_bs_" + str(bs) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_optim_" + optimizer + "_iid_" + str(iid) + "_ann_" + str(ann_baseline) + "_straggler_" + str(straggler_prob) + "_nobn_" + str(no_batchnorm) + "_grad_noise_" + str(grad_noise_stdev)
                    try:
                        file_name = "rog_scipt" + str(time.time()) + ".sh"
                        os.makedirs(dir_name + exp_name)
                        generate_script(file_name, gpu_type = gpu_type, grace_partition = grace_partition, exp_name = exp_name, dataset = dataset, num_classes = num_classes, model = model, iid = iid, optimizer = optimizer, bs = bs, lr = lr, lr_reduce = lr_reduce, epochs = epochs, local_ep = local_ep, eval_every = eval_every, num_users = num_users, frac = frac, gpu = gpu, timesteps = timesteps, dir_name = dir_name, ann_baseline = ann_baseline, straggler_prob = straggler_prob, no_batchnorm = no_batchnorm, grad_noise_stdev = grad_noise_stdev)
                        os.system("sbatch " + file_name)
                    except:
                        print("Error: Directory exists, it is possible that you are overwriting an experiment", dir_name)

                    iid = True
                    num_users = comb[0]
                    frac = comb[1]
                    exp_name = "fed_snn_bntt_nc_" + str(num_users) + "_frac_" + str(frac) + "_bs_" + str(bs) + "_lr_" + str(lr) + "_epochs_" + str(epochs) + "_optim_" + optimizer + "_iid_" + str(iid) + "_ann_" + str(ann_baseline) + "_straggler_" + str(straggler_prob) + "_nobn_" + str(no_batchnorm) + "_grad_noise_" + str(grad_noise_stdev)
                    try:
                        file_name = "rog_scipt" + str(time.time()) + ".sh"
                        os.makedirs(dir_name + exp_name)
                        generate_script(file_name, gpu_type = gpu_type, grace_partition = grace_partition, exp_name = exp_name, dataset = dataset, num_classes = num_classes, model = model, iid = iid, optimizer = optimizer, bs = bs, lr = lr, lr_reduce = lr_reduce, epochs = epochs, local_ep = local_ep, eval_every = eval_every, num_users = num_users, frac = frac, gpu = gpu, timesteps = timesteps, dir_name = dir_name, ann_baseline = ann_baseline, straggler_prob = straggler_prob, no_batchnorm = no_batchnorm, grad_noise_stdev = grad_noise_stdev)
                        os.system("sbatch " + file_name)
                    except:
                        print("Error: Directory exists, it is possible that you are overwriting an experiment", dir_name)

if __name__ == "__main__":
    main()
