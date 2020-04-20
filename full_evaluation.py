import os
import tensorflow as tf
from importlib.machinery import SourceFileLoader
import argparse
import numpy as np
from phiseg_test_quantitative import main as dist_eval
from phiseg_test_predictions import main as dsc_eval


def report(name, array):
    print(f'{name:s}:\t\t{np.mean(array):.6f} +- {np.std(array):.6f}')


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
            'detunet_4annot',
            'phiseg_7_5_1annot',
            'phiseg_7_5_4annot',
            'probunet_1annot',
            'probunet_4annot',
            'proposed_1annot',
            'proposed_4annot']

    for exp in exps:
        model_path = os.path.join(base_exp_path, exp)
        config_file = os.path.join(base_config_path, exp + '.py')
        config_module = config_file.split('/')[-1].rstrip('.py')
        exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()
        print(exp)
        dsc_path = os.path.join(model_path, 'dice_%s.npz' % model_selection)
        if not os.path.exists(dsc_path):
            dsc_eval(model_path, exp_config=exp_config, model_selection=model_selection)
            tf.reset_default_graph()

        dsc = np.load(dsc_path)['arr_0'][:, 1]
        report('DSC', dsc)

        if exp in ['detunet_1annot', 'detunet_4annot']:
            continue

        ged_path = os.path.join(model_path, 'ged%s_%s.npz' % (str(num_samples), model_selection))
        ncc_path = os.path.join(model_path, 'ncc%s_%s.npz' % (str(num_samples), model_selection))

        if not (os.path.exists(ged_path) and os.path.exists(ncc_path)):
            dist_eval(model_path, exp_config=exp_config, model_selection=model_selection)
            tf.reset_default_graph()

        ged = np.load(ged_path)['arr_0']
        report('GED', ged)
        ncc = np.load(ncc_path)['arr_0']
        report('NCC', ncc)
