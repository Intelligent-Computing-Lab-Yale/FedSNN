import os
import itertools
import pdb
import argparse
import time

def write_sbatch_conf(f, exp_name = "default", grace_partition = "pi_panda", dir_name = "./"):
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name=' + exp_name + '\n')
    f.write('#SBATCH --ntasks=1 --nodes=1\n')
    f.write('#SBATCH --partition=' + grace_partition + '\n')
    f.write('#SBATCH --mem=256G\n')
    f.write('#SBATCH --cpus-per-task=8\n')
    f.write('#SBATCH --time=24:00:00\n')
    f.write('#SBATCH --output=' + dir_name + exp_name + '.log\n')
    f.write('module load miniconda\n')
    f.write('conda activate ddd20\n')

def generate_script(file_name, exp_name = "default", grace_partition = "pi_panda", origin_dir = "data/fordfocus", out_dir="out_dir", to_do_file = "jul16/rec1500220388.hdf5"):
    in_full_file_prefix = origin_dir + "/" + os.path.splitext(to_do_file)[0]
    base_id = os.path.basename(in_full_file_prefix)
    out_full_file_prefix = out_dir + "/" + base_id
    
    f = open(file_name, 'w', buffering = 1)
    write_sbatch_conf(f, exp_name, grace_partition, out_dir + "/logs/")
    s = 'echo "Working on ' + out_full_file_prefix + '"\n'
    f.write(s)
    s = "ipython ./export.py -- " + in_full_file_prefix + ".hdf5 --binsize 0.01 --export_aps 1 --export_dvs 0 --out_file " + out_full_file_prefix + "_frames.hdf5\n"
    f.write(s)
    # s = "ipython ./export.py -- " + in_full_file_prefix + ".hdf5 --binsize 0.01 --export_aps 0 --export_dvs 1 --out_file " + out_full_file_prefix + "_with_timesteps.hdf5 --split_timesteps --timesteps 5\n"
    # f.write(s)
    s = "ipython ./export.py -- " + in_full_file_prefix + ".hdf5 --binsize 0.01 --export_aps 0 --export_dvs 1 --out_file " + out_full_file_prefix + "_dvs_accum_frames.hdf5\n"
    f.write(s)

    # Prepare and resize
    # ------------ Prepare APS -------------#
    s = "ipython ./prepare_cnn_data.py -- --filename " + out_full_file_prefix + "_frames.hdf5 --rewrite 1 --skip_mean_std 1"
    f.write(s)
    # ----------- Prepare timestep split DVS ------- #
    # s = "ipython ./prepare_cnn_data.py -- --filename " + out_full_file_prefix + "_with_timesteps.hdf5 --rewrite 1 --skip_mean_std 1 --split_timesteps --timesteps 5"
    # ----------- Prepare accumulated DVS ----------- #
    # f.write(s)
    s = "ipython ./prepare_cnn_data.py -- --filename " + out_full_file_prefix + "_dvs_accum_frames.hdf5 --rewrite 1 --skip_mean_std 1"
    f.write(s)

    f.close()

def main():
    # Constants
    grace_partition = "bigmem"
    origin_dir = "data/fordfocus/"
    bin_size = "10ms"
    result_dir = "processed_dataset/"

    day_files = ['jul16/rec1500220388.hdf5', 'jul18/rec1500383971.hdf5', 'jul18/rec1500402142.hdf5', 'jul28/rec1501288723.hdf5', 'jul29/rec1501349894.hdf5', 'aug01/rec1501614399.hdf5', 'aug08/rec1502241196.hdf5', 'aug15/rec1502825681.hdf5', 'jul02/rec1499023756.hdf5', 'jul05/rec1499275182.hdf5', 'jul08/rec1499533882.hdf5', 'jul16/rec1500215505.hdf5', 'jul17/rec1500314184.hdf5', 'jul17/rec1500329649.hdf5', 'aug05/rec1501953155.hdf5']
    night_files = ['jul09/rec1499656391.hdf5', 'jul09/rec1499657850.hdf5', 'aug01/rec1501649676.hdf5', 'aug01/rec1501650719.hdf5', 'aug05/rec1501994881.hdf5', 'aug09/rec1502336427.hdf5', 'aug09/rec1502337436.hdf5', 'jul01/rec1498946027.hdf5', 'aug01/rec1501651162.hdf5', 'jul02/rec1499025222.hdf5', 'aug09/rec1502338023.hdf5', 'aug09/rec1502338983.hdf5', 'aug09/rec1502339743.hdf5', 'jul01/rec1498949617.hdf5', 'aug12/rec1502599151.hdf5']

    out_dir = result_dir + "day"

    for to_do_file in day_files[0:1]:
        file_name = "rog_scipt" + str(time.time()) + ".sh"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            os.makedirs(out_dir+"/logs")
        exp_name = "ddd_preprocessing_" + os.path.splitext(os.path.basename(to_do_file))[0]
        generate_script(file_name, exp_name, grace_partition, origin_dir, out_dir, to_do_file)
        os.system("sbatch " + file_name)
    
    out_dir = result_dir + "night"

    for to_do_file in night_files[0:1]:
        file_name = "rog_scipt" + str(time.time()) + ".sh"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            os.makedirs(out_dir+"/logs")
        exp_name = "ddd_preprocessing_" + os.path.splitext(os.path.basename(to_do_file))[0]
        generate_script(file_name, exp_name, grace_partition, origin_dir, out_dir, to_do_file)
        os.system("sbatch " + file_name)

if __name__ == "__main__":
    main()
