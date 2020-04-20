# Get classification metrics for a trained classifier model
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

from phiseg.model_zoo import likelihoods
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
from medpy.metric import dc

import config.system as sys_config
from phiseg.phiseg_model import phiseg
import utils

if not sys_config.running_on_gpu_host:
    import matplotlib.pyplot as plt

import logging
from data.data_switch import data_switch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}


def main(model_path, exp_config, model_selection='latest'):

    # Get Data
    phiseg_model = phiseg(exp_config=exp_config)
    phiseg_model.load_weights(model_path, type=model_selection)

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    # Run predictions in an endless loop
    dice_list = []

    num_samples = 1 if exp_config.likelihood is likelihoods.det_unet2D else 100

    for ii, batch in enumerate(data.test.iterate_batches(1)):

        if ii % 10 == 0:
            logging.info("Progress: %d" % ii)

        # print(ii)

        x, y = batch

        y_ = np.squeeze(phiseg_model.predict(x, num_samples=num_samples))

        per_lbl_dice = []
        per_pixel_preds = []
        per_pixel_gts = []

        for lbl in range(exp_config.nlabels):

            binary_pred = (y_ == lbl) * 1
            binary_gt = (y == lbl) * 1

            if np.sum(binary_gt) == 0 and np.sum(binary_pred) == 0:
                per_lbl_dice.append(1)
            elif np.sum(binary_pred) > 0 and np.sum(binary_gt) == 0 or np.sum(binary_pred) == 0 and np.sum(binary_gt) > 0:
                logging.warning('Structure missing in either GT (x)or prediction. ASSD and HD will not be accurate.')
                per_lbl_dice.append(0)
            else:
                per_lbl_dice.append(dc(binary_pred, binary_gt))

        dice_list.append(per_lbl_dice)

        per_pixel_preds.append(y_.flatten())
        per_pixel_gts.append(y.flatten())

    dice_arr = np.asarray(dice_list)

    mean_per_lbl_dice = dice_arr.mean(axis=0)

    logging.info('Dice')
    logging.info(mean_per_lbl_dice)
    logging.info(np.mean(mean_per_lbl_dice))
    logging.info('foreground mean: %f' % (np.mean(mean_per_lbl_dice[1:])))

    np.savez(os.path.join(model_path, 'dice_%s.npz' % model_selection), dice_arr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a network on the test dataset")
    parser.add_argument("--exp-path", type=str, help="Path to experiment folder")
    parser.add_argument("--config-file", type=str, help="Path to config")
    parser.add_argument("--device", type=str, help="device for computation")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_path = args.exp_path
    config_file = args.config_file
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config, do_plots=False)

