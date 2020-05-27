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
import SimpleITK as sitk
import math
import pickle
from scipy.misc import logsumexp

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def nanstderr(array):
    return np.nanstd(array) / np.sqrt(np.sum(np.logical_not(np.isnan(array))))


def update_output_dataframe(exp, output_dataframe, exp_results, det_exp_results, num_classes):
    exp_results['normalised_entropy'] = exp_results['entropy'] / (128 ** 2 * math.log(num_classes))

    for column in ['ged', 'ncc', 'entropy', 'diversity', 'normalised_entropy', 'ece', 'unweighted_ece',
                   'loglikelihood']:
        output_dataframe.loc[exp, column + '_mean'] = np.nanmean(exp_results[column])
        output_dataframe.loc[exp, column + '_stderr'] = nanstderr(exp_results[column])

    for c in range(1, num_classes):
        # after looking at the data sets are no in fact per expert as there are far more than 4 experts so it makes
        # sense to aggregate
        dsc = np.concatenate(exp_results['dsc'][..., c])
        positives = np.concatenate(exp_results['presence'][..., c])
        negatives = np.logical_not(positives)
        false_positives = np.logical_and(dsc == 0., negatives)
        false_negatives = np.logical_and(dsc == 0., positives)
        output_dataframe.loc[exp, f'dsc_c_{c:d}_mean'] = np.nanmean(dsc)
        output_dataframe.loc[exp, f'dsc_c_{c:d}_stderr'] = nanstderr(dsc)
        output_dataframe.loc[exp, f'dsc_c_{c:d}_where_lesion_mean'] = np.nanmean(dsc[positives])
        output_dataframe.loc[exp, f'dsc_c_{c:d}_where_lesion_stderr'] = nanstderr(dsc[positives])
        output_dataframe.loc[exp, f'fpr_c_{c:d}'] = np.sum(false_positives) / np.sum(negatives)
        output_dataframe.loc[exp, f'fnr_c_{c:d}'] = np.sum(false_negatives) / np.sum(positives)
        output_dataframe.loc[exp, f'positives_c_{c:d}'] = np.sum(positives)
        output_dataframe.loc[exp, f'negatives_c_{c:d}'] = np.sum(negatives)
    sample_gain = (exp_results['sample_dsc'] - np.expand_dims(det_exp_results['dsc'], 1))[..., 1:]
    output_dataframe.loc[exp, f'median_gain_mean'] = np.nanmean(np.nanmedian(sample_gain, axis=1))
    return output_dataframe


def summarize_results(base_exp_path, exps, num_classes=2, model_selection='latest', num_samples=100, mode=False):
    output_dataframe = pd.DataFrame(index=exps)
    for exp in exps:
        exp_path = get_output_path(os.path.join(base_exp_path, exp), num_samples, model_selection, mode) + '.pickle'
        with open(exp_path, 'rb') as f:
            exp_results = pickle.load(f)
        det = 0 if '1annot' in exp else 5
        det_exp_path = get_output_path(os.path.join(base_exp_path, exps[det]), num_samples, model_selection,
                                       mode) + '.pickle'
        with open(det_exp_path, 'rb') as f:
            det_exp_results = pickle.load(f)
        output_dataframe = update_output_dataframe(exp, output_dataframe, exp_results, det_exp_results, num_classes)
    return output_dataframe


def report_array(array, name):
    print(f'{name:s}:\n{np.mean(array):.6f} +- {np.std(array):.6f}')


def report_dataframe(dataframe, num_classes=2, num_experts=4):
    report_array(np.array(dataframe['ged']), 'ged')
    report_array(np.array(dataframe['ncc']), 'ncc')
    report_array(np.array(dataframe['entropy']), 'entropy')
    report_array(np.array(dataframe['diversity']), 'diversity')

    for c in range(1, num_classes):
        for e in range(num_experts):
            key = f'_c_{c:d}_e_{e:d}'
            dsc = dataframe['dsc' + key]
            presence = dataframe['presence' + key]
            report_array(dsc, 'dsc' + key)
            report_array(dsc[presence], 'dsc where lesion' + key)
            dsc[np.isnan(dsc)] = 1.
            report_array(dsc, 'Alt_' + key)


def make_dataframe(metrics):
    data_dict = {key: np.array(metric) for key, metric in metrics.items()}
    dsc = data_dict['dsc']
    presence = data_dict['presence']
    for e in range(dsc.shape[1]):
        for c in range(dsc.shape[-1]):
            data_dict.update({f'dsc_c_{c:d}_e_{e:d}': dsc[:, e, c]})
            data_dict.update({f'presence_c_{c:d}_e_{e:d}': presence[:, e, c]})
    data_dict.pop('dsc')
    data_dict.pop('presence')
    return pd.DataFrame(data_dict)


def calc_dsc(image_0, image_1):
    if np.sum(image_0) == 0 and np.sum(image_1) == 0:
        return np.nan
    else:
        return dc(image_1, image_0)


def get_output_path(model_path, num_samples, model_selection, mode):
    if not mode:
        return os.path.join(model_path, f'test_results_{num_samples:d}_samples_{model_selection:s}')
    else:
        return os.path.join(model_path, f'test_results_{num_samples:d}_samples_{model_selection:s}_mode')


class ImageSaver(object):
    def __init__(self, output_path, samples_to_keep=20):
        self.output_path = output_path
        self.samples_to_keep = samples_to_keep
        self.df = pd.DataFrame()

    def save_image(self, image, id_, name, dtype):
        path = os.path.join(self.output_path, id_ + name + '.nii.gz')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(image.astype(dtype)), path)
        return path

    def __call__(self, id_, image, targets, prediction, samples):
        self.df.loc[id_, 'image'] = self.save_image(image, id_, 'image', np.float32)
        for i, target in enumerate(targets):
            self.df.loc[id_, f'target_{i:d}'] = self.save_image(target, id_, f'target_{i:d}.nii.gz', np.uint8)

        self.df.loc[id_, 'image'] = self.save_image(prediction, id_, 'prediction.nii.gz', np.float32)
        samples_to_keep = min(self.samples_to_keep, len(samples))
        for i, sample in enumerate(samples[:samples_to_keep]):
            self.df.loc[id_, f'sample_{i:d}'] = self.save_image(sample, id_, f'sample_{i:d}.nii.gz', np.uint8)

    def close(self):
        self.df.to_csv(os.path.join(self.output_path, 'sampling.csv'), index=False)


def calculate_log_likelihood(targets, sample_prob_maps):
    m = sample_prob_maps.shape[0]
    targets = np.expand_dims(np.stack((targets, np.logical_not(targets)), -1), 1)
    sample_prob_maps = np.expand_dims(sample_prob_maps, 0)
    return logsumexp(np.sum(targets * np.log(sample_prob_maps + 1e-10), axis=(2, 3, 4)), axis=1) - np.log(m)


def calculate_expert_diversity(exp_config):
    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)
    diversity = []
    for ii in tqdm(range(data.test.images.shape[0])):
        targets = data.test.labels[ii, ...].transpose((2, 0, 1))
        ged_, diversity_ = utils.generalised_energy_distance(targets, targets, exp_config.nlabels - 1,
                                                             range(1, exp_config.nlabels))
        diversity.append(diversity_)
    diversity = np.array(diversity)
    print(f'{np.mean(diversity):.6f} +- {nanstderr(diversity):.6f}')


def calc_class_wise_expected_calibration_error(targets, prob_map, num_classes, num_bins):
    bins = np.linspace(0, 1, num_bins + 1)
    prob_map = np.transpose(prob_map, axes=(-1,) + tuple(range(len(prob_map.shape) - 1)))
    class_proportions = []
    total = []
    confidence = []
    for j in range(len(bins) - 1):
        start = bins[j]
        end = bins[j + 1] + 1 if j == len(bins) - 2 else bins[j + 1]
        ind = np.logical_and(prob_map >= start, prob_map < end)
        confidence.append(np.stack([np.nanmean(prob_map[c, ind[c]]) for c in range(num_classes)]))
        total.append(np.sum(ind, axis=(-1, -2)))
        class_incidence = np.stack([np.logical_and(targets == c, ind[c]) for c in range(num_classes)])
        class_proportions.append(np.sum(class_incidence, axis=(-1, -2)))

    confidence = np.array(confidence)
    total = np.array(total)
    class_proportions = np.nanmean(np.array(class_proportions) / np.expand_dims(total, -1), axis=-1)
    ece = np.nansum(np.abs(confidence - class_proportions) * total, axis=0) / np.nansum(total, axis=0)
    unweighted_ece = np.nanmean(np.abs(confidence - class_proportions), axis=0)
    return ece, unweighted_ece


def test(model_path, exp_config, model_selection='latest', num_samples=100, overwrite=False, mode=False):
    output_path = get_output_path(model_path, num_samples, model_selection, mode) + '.pickle'
    if os.path.exists(output_path) and not overwrite:
        return
    image_saver = ImageSaver(os.path.join(model_path, 'samples'))
    tf.reset_default_graph()
    phiseg_model = phiseg(exp_config=exp_config)
    phiseg_model.load_weights(model_path, type=model_selection)

    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    metrics = {key: [] for key in
               ['dsc', 'presence', 'ged', 'ncc', 'entropy', 'diversity', 'sample_dsc', 'ece', 'unweighted_ece',
                'loglikelihood']}

    num_samples = 1 if exp_config.likelihood is likelihoods.det_unet2D else num_samples

    for ii in tqdm(range(data.test.images.shape[0])):
        image = data.test.images[ii, ...].reshape([1] + list(exp_config.image_size))
        targets = data.test.labels[ii, ...].transpose((2, 0, 1))

        feed_dict = {phiseg_model.training_pl: False,
                     phiseg_model.x_inp: np.tile(image, [num_samples, 1, 1, 1])}

        prob_maps = phiseg_model.sess.run(phiseg_model.s_out_eval_sm, feed_dict=feed_dict)
        samples = np.argmax(prob_maps, axis=-1)
        probability = np.mean(prob_maps, axis=0) + 1e-10
        metrics['entropy'].append(float(np.sum(-probability * np.log(probability))))
        if mode:
            prediction = np.round(np.mean(np.argmax(prob_maps, axis=-1), axis=0)).astype(np.int64)
        else:
            if 'proposed' not in exp_config.experiment_name:
                prediction = np.argmax(np.sum(prob_maps, axis=0), axis=-1)
            else:
                mean = phiseg_model.sess.run(phiseg_model.dist_eval.loc, feed_dict=feed_dict)[0]
                mean = np.reshape(mean, image.shape[:-1] + (2,))
                prediction = np.argmax(mean, axis=-1)

        metrics['loglikelihood'].append(calculate_log_likelihood(targets, prob_maps))
        # calculate DSC per expert
        metrics['dsc'].append(
            [[calc_dsc(target == i, prediction == i) for i in range(exp_config.nlabels)] for target in targets])
        metrics['presence'].append([[np.any(target == i) for i in range(exp_config.nlabels)] for target in targets])

        metrics['sample_dsc'].append([[[calc_dsc(target == i, sample == i) for i in range(exp_config.nlabels)]
                                       for target in targets] for sample in samples])

        # ged and diversity
        ged_, diversity_ = utils.generalised_energy_distance(samples, targets, exp_config.nlabels - 1,
                                                             range(1, exp_config.nlabels))
        metrics['ged'].append(ged_)
        metrics['diversity'].append(diversity_)
        # NCC
        targets_one_hot = utils.to_one_hot(targets, exp_config.nlabels)
        metrics['ncc'].append(utils.variance_ncc_dist(prob_maps, targets_one_hot)[0])
        prob_map = np.mean(prob_maps, axis=0)
        ece, unweighted_ece = calc_class_wise_expected_calibration_error(targets, prob_map, 2, 10)
        metrics['ece'].append(ece)
        metrics['unweighted_ece'].append(unweighted_ece)
        image_saver(str(ii) + '/', image[0, ..., 0], targets, prediction, samples)

    metrics = {key: np.array(metric) for key, metric in metrics.items()}
    with open(output_path, 'wb') as f:
        pickle.dump(metrics, f)
    image_saver.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, help="device for computation")
    parser.add_argument("--model-selection", type=str, help="model selection criterion", default='latest')
    parser.add_argument("--num-samples", type=int, help="number of samples for distribution evaluation", default=100)
    parser.add_argument("--overwrite", type=bool, help="overwrite previous results", default=False)
    parser.add_argument("--mode", type=bool, help="whether to use mode as prediction", default=False)
    parser.add_argument("--seed", type=int, help="random seed", default=10)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

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

    for i, exp in enumerate(exps):
        model_path = os.path.join(base_exp_path, exp)
        config_file = os.path.join(base_config_path, exp + '.py')
        config_module = config_file.split('/')[-1].rstrip('.py')
        exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()
        if i == 0:
            calculate_expert_diversity(exp_config)
        test(model_path, exp_config, model_selection, num_samples, args.overwrite, args.mode)

    output_dataframe = summarize_results(base_exp_path, exps, 2, model_selection, num_samples, args.mode)
    output_path = get_output_path(base_exp_path, num_samples, model_selection, args.mode) + '.csv'
    output_dataframe.to_csv(os.path.join(base_exp_path, output_path))
