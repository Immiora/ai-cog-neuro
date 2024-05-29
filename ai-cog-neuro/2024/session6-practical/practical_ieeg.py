import numpy as np
import mne
import mne_bids
import openneuro as on

## Download data
bids_dir = 'data/bids'
subject = 'sub-07'
on.download(dataset='ds003688', target_dir=bids_dir, include=subject)

## Specify data
acquisition = 'clinical'
task = 'film'
datatype = 'ieeg'
session = 'iemu'

## Load channels
import pandas as pd
channels_path = mne_bids.BIDSPath(subject='07',
                                    session=session,
                                    suffix='channels',
                                    extension='.tsv',
                                    datatype=datatype,
                                    task=task,
                                    acquisition=acquisition,
                                    root=bids_dir)
channels = pd.read_csv(str(channels_path.match()[0]), sep='\t', header=0, index_col=None)

## Select channels
data_path = mne_bids.BIDSPath(subject='07',
                                    session=session,
                                    suffix='ieeg',
                                    extension='.vhdr',
                                    datatype=datatype,
                                    task=task,
                                    acquisition=acquisition,
                                    root=bids_dir)
raw = mne.io.read_raw_brainvision(str(data_path.match()[0]), scale=1.0, preload=False, verbose=True)
raw.set_channel_types({ch_name: str(x).lower()
                if str(x).lower() in ['ecog', 'seeg', 'eeg'] else 'misc'
                                for ch_name, x in zip(raw.ch_names, channels['type'].values)})
raw.drop_channels([raw.ch_names[i] for i, j in enumerate(raw.get_channel_types()) if j == 'misc'])

## Drop bad channels
bad_channels = channels['name'][(channels['type'].isin(['ECOG', 'SEEG'])) & (channels['status'] == 'bad')].tolist()
raw.info['bads'].extend([ch for ch in bad_channels])
raw.drop_channels(raw.info['bads'])

## Load data
raw.load_data()

## Plot data

## Report events / annotations

## Apply notch filter: remove line noise
raw.notch_filter(freqs=np.arange(50, 251, 50))

## Apply common average reference
raw_car, _ = mne.set_eeg_reference(raw.copy(), 'average')

## Extract a high frequency component of the signal

# with hilbert transform
#gamma = raw_car.copy().filter(60, 120).apply_hilbert(envelope=True).get_data().T

# with a wavelet transform
temp = mne.time_frequency.tfr_array_morlet(np.expand_dims(raw_car.copy()._data, 0), # (n_epochs, n_channels, n_times)
                                                     sfreq=raw.info['sfreq'],
                                                     freqs=np.arange(70, 170),
                                                     verbose=True,
                                                     n_cycles=4.,
                                                     n_jobs=1)
gamma = np.mean(np.abs(temp), 2).squeeze().T


## Load events
custom_mapping = {'Stimulus/music': 2, 'Stimulus/speech': 1,
                  'Stimulus/end task': 5}  # 'Stimulus/task end' in laan
events, event_id = mne.events_from_annotations(raw_car, event_id=custom_mapping,
                                                         use_rounding=False)

raw_car.plot(events=events, start=0, duration=180, color='gray', event_color={2: 'g', 1: 'r'}, bgcolor='w')

## Crop to start and end of the task
gamma_cropped = gamma[events[0, 0]:events[-1, 0]]

## Resample to a lower sampling rate
def resample(x, sr1, sr2, axis=0):
    '''
    Resample signal

    :param x: ndarray, time x channels
    :param sr1: float: target sampling rate
    :param sr2: float: source sampling rate
    :param axis: axis of array to apply function to
    :return:
    '''
    from fractions import Fraction
    from scipy.signal import resample_poly
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)


gamma_resampled = resample(gamma_cropped, 25, int(raw.info['sfreq']))

## Save the processed data
from pathlib import Path
fname = Path('data' / 'output' / 'ieeg'/ 'processed_gamma.npy')
np.save(fname, gamma_resampled)

## Plot the data you have with
# from matplotlib import pyplot as plt
# plt.plot(signal)

## Try smoothing data to make it less noisy
def smooth_signal_1d(y, n):
    '''
    Smooth 1d signal

    :param y: ndarray: signal
    :param n: size of smoothing
    :return: smoothed signal
    '''
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(y, axis=0, size=n)

## Try PCA to reduce the dimensionality

## Split the data into cross-validation folds

## Train a classifier to predict labels: 1 for speech and 0 for music

## Try different classifiers and parameters
