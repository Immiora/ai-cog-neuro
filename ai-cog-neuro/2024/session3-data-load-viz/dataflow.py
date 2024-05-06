import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import mne
import subprocess
##
subprocess.run("git clone https://gin.g-node.org/sappelhoff/eeg_matchingpennies.git")
##
subprocess.run("wget -P eeg_matchingpennies/sub-05/eeg/ https://gin.g-node.org/sappelhoff/eeg_matchingpennies/raw/master/sub-05/eeg/sub-05_task-matchingpennies_eeg.eeg")


###
import mne_bids