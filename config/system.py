# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.


project_root = <PROJECT_ROOT>
local_hostnames = [<HOST_NAME>]  # used to check if on cluster or not
log_root = <LOG_ROOT>

##################################################################################

running_on_gpu_host = True if socket.gethostname() not in local_hostnames else False
