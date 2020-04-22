import os
import numpy as np
import tensorflow as tf
import pandas as pd
from importlib.machinery import SourceFileLoader
import argparse
from medpy.metric import dc
from tqdm import tqdm
import utils
from phiseg.phiseg_model import phiseg
from phiseg.model_zoo import likelihoods
from data.data_switch import data_switch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def summarize_results(base_exp_path, exps, model_selection='latest', num_samples=100):
    output_dataframe = pd.DataFrame(index=exps)
    for exp in exps:
        csv_path = os.path.join(base_exp_path, exp, f'test_results_{num_samples:d}_samples_{model_selection:s}.csv')
        results = pd.read_csv(csv_path)
        for column in results.columns:
            if 'DSC' in column:
                alt_dsc = np.array(results[column])
                alt_dsc[np.isnan(alt_dsc)] = 1.
                results[column + '_alt'] = alt_dsc
        for column in results.columns:
            output_dataframe.loc[exp, column + '_mean'] = np.nanmean(results[column])
            output_dataframe.loc[exp, column + '_std'] = np.nanstd(results[column])
    return output_dataframe


def report_array(array, name):
    print(f'{name:s}:\t\t{np.mean(array):.6f} +- {np.std(array):.6f}')


def report_dataframe(dataframe, num_classes=2, num_experts=4):
    report_array(np.array(dataframe['GED']), 'GED')
    report_array(np.array(dataframe['NCC']), 'NCC')

    for c in range(1, num_classes):
        for e in range(num_experts):
            key = f'DSC_c_{c:d}_e_{e:d}'
            dsc = dataframe[key]
            report_array(dsc, key)
            dsc[np.isnan(dsc)] = 1.
            report_array(dsc, 'Alt_' + key)


def make_dataframe(ged, ncc, dsc):
    dsc = np.array(dsc)
    ged = np.array(ged)
    ncc = np.array(ncc)
    data_dict = {'GED': ged, 'NCC': ncc}

    for e in range(dsc.shape[1]):
        for c in range(dsc.shape[-1]):
            data_dict.update({f'DSC_c_{c:d}_e_{e:d}': dsc[:, e, c]})

    return pd.DataFrame(data_dict)


def calc_dsc(image_0, image_1):
    if np.sum(image_0) == 0 and np.sum(image_1) == 0:
        return np.nan
    else:
        return dc(image_1, image_0)


def test(model_path, exp_config, model_selection='latest', num_samples=100):
    output_path = os.path.join(model_path, f'test_results_{num_samples:d}_samples_{model_selection:s}.csv')
    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    tf.reset_default_graph()
    phiseg_model = phiseg(exp_config=exp_config)
    phiseg_model.load_weights(model_path, type=model_selection)

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    dsc = []
    ged = []
    ncc = []

    num_samples = 1 if exp_config.likelihood is likelihoods.det_unet2D else num_samples

    for ii in tqdm(range(data.test.images.shape[0])):
        image = data.test.images[ii, ...].reshape([1] + list(exp_config.image_size))
        targets = data.test.labels[ii, ...].transpose((2, 0, 1))

        feed_dict = {phiseg_model.training_pl: False,
                     phiseg_model.x_inp: np.tile(image, [num_samples, 1, 1, 1])}

        probs = phiseg_model.sess.run(phiseg_model.s_out_eval_sm, feed_dict=feed_dict)
        samples = np.argmax(probs, axis=-1)
        if 'proposed' not in exp_config.experiment_name:
            prediction = np.argmax(np.sum(probs, axis=0), axis=-1)
        else:
            mean = phiseg_model.sess.run(phiseg_model.dist_eval.loc, feed_dict=feed_dict)[0]
            mean = np.reshape(mean, image.shape[:-1] + (2,))
            prediction = np.argmax(mean, axis=-1)

        # calculate DSC per expert
        dsc.append([[calc_dsc(target == i, prediction == i) for i in range(exp_config.nlabels)] for target in targets])

        if 'detunet' in exp_config.experiment_name:
            ged.append(0.)
            ncc.append(0.)
        else:
            targets_one_hot = utils.convert_batch_to_onehot(targets, exp_config.nlabels)
            ged.append(utils.generalised_energy_distance(samples, targets, nlabels=exp_config.nlabels - 1,
                                                         label_range=range(1, exp_config.nlabels)))
            ncc.append(utils.variance_ncc_dist(probs, targets_one_hot)[0])
    dataframe = make_dataframe(ged, ncc, dsc)
    dataframe.to_csv(output_path, index=False)
    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, help="device for computation")
    parser.add_argument("--model-selection", type=str, help="model selection criterion", default='latest')
    parser.add_argument("--num-samples", type=int, help="number of samples for distribution evaluation", default=100)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    model_selection = args.model_selection
    num_samples = args.num_samples

    base_exp_path = '/vol/biomedic/users/mm6818/Projects/variational_hydra/phiseg_jobs/lidc'
    base_config_path = 'phiseg/experiments'

    exps = ['detunet_1annot',
            'probunet_1annot',
            'phiseg_7_5_1annot',
            'proposed_diag_1annot',
            'proposed_1annot',
            'detunet_4annot',
            'probunet_4annot',
            'phiseg_7_5_4annot',
            'proposed_diag_4annot',
            'proposed_4annot']

    for exp in exps:
        model_path = os.path.join(base_exp_path, exp)
        config_file = os.path.join(base_config_path, exp + '.py')
        config_module = config_file.split('/')[-1].rstrip('.py')
        exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

        dataframe = test(model_path, exp_config, model_selection, num_samples)
        print(exp)
        report_dataframe(dataframe, num_classes=2, num_experts=4)

    output_dataframe = summarize_results(base_exp_path, exps, model_selection, num_samples)
    output_dataframe.to_csv(
        os.path.join(base_exp_path, f'test_results_{num_samples:d}_samples_{model_selection:s}.csv'))
