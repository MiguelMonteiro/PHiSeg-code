# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

at_biwi = True  # Are you running this code from the ETH Computer Vision Lab (Biwi)?

project_root = '/vol/biomedic/users/mm6818/Projects/variational_hydra/PHiSeg-code'
local_hostnames = ['battle']  # used to check if on cluster or not
log_root = '/vol/biomedic/users/mm6818/variational_hydra/PhiSeg-code/logs'

##################################################################################

running_on_gpu_host = True if socket.gethostname() not in local_hostnames else False


def setup_GPU_environment():
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    logging.info('Setting CUDA_VISIBLE_DEVICES variable...')
    logging.info('CUDA_VISIBLE_DEVICES is %s' % os.environ['CUDA_VISIBLE_DEVICES'])

