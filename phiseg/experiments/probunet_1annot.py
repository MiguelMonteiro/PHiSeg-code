from phiseg.model_zoo import likelihoods, posteriors, priors
import tensorflow as tf
from tfwrapper import normalisation as tfnorm

experiment_name = 'probunet_1annot'
log_dir_name = 'lidc'

# architecture
posterior = posteriors.prob_unet2D
likelihood = likelihoods.prob_unet2D
prior = priors.prob_unet2D
layer_norm = tfnorm.batch_norm  # No layer normalisation!
use_logistic_transform = False

latent_levels = 1
resolution_levels = 7
n0 = 32
zdim0 = 6

# Data settings
data_identifier = 'lidc'
preproc_folder = '/vol/biomedic/users/mm6818/Projects/variational_hydra/data/LIDC_2D_PHiSeg'
data_root = '/vol/biomedic/users/mm6818/data/LIDC/prob_unet/data_lidc.pickle'
dimensionality_mode = '2D'
image_size = (128, 128, 1)
nlabels = 2
num_labels_per_subject = 4

augmentation_options = {'do_flip_lr': True,
                        'do_flip_ud': True,
                        'do_rotations': True,
                        'do_scaleaug': True,
                        'nlabels': nlabels}

# training
optimizer = tf.train.AdamOptimizer
lr_schedule_dict = {0: 1e-3}
# lr_schedule_dict = {0: 1e-4, 80000: 0.5e-4, 160000: 1e-5, 240000: 0.5e-6} #  {0: 1e-3}
deep_supervision = True
batch_size = 12
num_iter = 500000
annotator_range = [0]  # which annotators to actually use for training

# losses
KL_divergence_loss_weight = 1.0
exponential_weighting = True

residual_multinoulli_loss_weight = 1.0

# monitoring
do_image_summaries = True
rescale_RGB = False
validation_frequency = 500
validation_samples = 16
num_validation_images = 100 #'all'
tensorboard_update_frequency = 100

