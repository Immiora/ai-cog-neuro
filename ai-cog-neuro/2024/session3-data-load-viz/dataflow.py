import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
import mne_bids

## specify bids information to load the data
bids_root = "eeg_matchingpennies"
bids_path = mne_bids.BIDSPath(root=bids_root, subject="05",
                              task="matchingpennies", datatype="eeg", suffix="eeg")
eeg_raw = mne_bids.read_raw_bids(bids_path, verbose=True)
eeg_raw.load_data()
eeg_raw.info

## raw time-domain plot
eeg_raw.plot(start=0,duration=30)
#eeg_raw.plot(start=0,duration=len(eeg_raw)/eeg_raw.info['sfreq'])
#eeg_raw.plot(start=0,duration=30, highpass=.1)

## raw frequency-domain plot
eeg_raw.compute_psd(fmax=500).plot()

## summary of the data
eeg_raw.describe()


## preprocessing
# high-pass filter
eeg_raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)

# epoch data based on events
id_dict = {"raised-left/match-true": 1, "raised-right/match-false": 1}
events, event_ids = mne.events_from_annotations(eeg_raw, event_id=id_dict, verbose=False)
tmin, tmax = 0, 4  # in s
baseline = None
eeg_epochs = mne.Epochs(
    eeg_raw,
    events=events,
    event_id=event_ids,
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    verbose=False)

eeg_epochs.plot(n_epochs=10, picks=['eeg'])