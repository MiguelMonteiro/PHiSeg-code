# Get classification metrics for a trained classifier model
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse

import config.system as sys_config
from phiseg.phiseg_model import phiseg
import utils

import logging
from data.data_switch import data_switch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}


def main(model_path, exp_config, model_selection='latest'):
    n_samples = 100

    # Get Data
    phiseg_model = phiseg(exp_config=exp_config)
    phiseg_model.load_weights(model_path, type=model_selection)

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    N = data.test.images.shape[0]

    ged_list = []
    ncc_list = []

    for ii in range(N):

        if ii % 10 == 0:
            logging.info("Progress: %d" % ii)

        x_b = data.test.images[ii, ...].reshape([1] + list(exp_config.image_size))
        s_b = data.test.labels[ii, ...]

        x_b_stacked = np.tile(x_b, [n_samples, 1, 1, 1])

        feed_dict = {}
        feed_dict[phiseg_model.training_pl] = False
        feed_dict[phiseg_model.x_inp] = x_b_stacked

        s_arr_sm = phiseg_model.sess.run(phiseg_model.s_out_eval_sm, feed_dict=feed_dict)
        s_arr = np.argmax(s_arr_sm, axis=-1)

        # s_arr = np.squeeze(np.asarray(s_list)) # num samples x X x Y
        s_b_r = s_b.transpose((2, 0, 1))  # num gts x X x Y
        s_b_r_sm = utils.to_one_hot(s_b_r, exp_config.nlabels)  # num gts x X x Y x nlabels

        ged = utils.generalised_energy_distance(s_arr, s_b_r, nlabels=exp_config.nlabels - 1,
                                                label_range=range(1, exp_config.nlabels))
        ged_list.append(ged)

        ncc = utils.variance_ncc_dist(s_arr_sm, s_b_r_sm)
        ncc_list.append(ncc)

    ged_arr = np.asarray(ged_list)
    ncc_arr = np.asarray(ncc_list)

    logging.info('-- GED: --')
    logging.info(np.mean(ged_arr))
    logging.info(np.std(ged_arr))

    logging.info('-- NCC: --')
    logging.info(np.mean(ncc_arr))
    logging.info(np.std(ncc_arr))

    np.savez(os.path.join(model_path, 'ged%s_%s.npz' % (str(n_samples), model_selection)), ged_arr)
    np.savez(os.path.join(model_path, 'ncc%s_%s.npz' % (str(n_samples), model_selection)), ncc_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a network on the test dataset")
    parser.add_argument("--exp-path", type=str, help="Path to experiment folder")
    parser.add_argument("--config-file", type=str, help="Path to config")
    parser.add_argument("--device", type=str, help="device for computation")
    args = parser.parse_args()

    base_path = sys_config.project_root
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_path = args.exp_path
    config_file = args.config_file
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config, do_plots=False)
