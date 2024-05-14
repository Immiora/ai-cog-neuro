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
eeg_raw.plot(start=0, duration=eeg_raw._data.shape[-1]/eeg_raw.info['sfreq'])

## raw frequency-domain plot
eeg_raw.compute_psd().plot()

## summary of the data
eeg_raw.describe()
eeg_raw.annotations.description

## preprocess
# high-pass filter
eeg_raw.filter(l_freq=.3, h_freq=70, fir_design="firwin", verbose=False)
# reference
eeg_raw, car = mne.set_eeg_reference(eeg_raw, 'average')
# slow potentials
eeg_raw.filter(l_freq=.5, h_freq=5, fir_design="firwin", verbose=False)
# downsample
eeg_raw.resample(sfreq=10)

## plot the psd again
eeg_raw.plot_psd(fmax=5)

## epoch data based on events
id_dict = {"raised-left/match-true": 0, "raised-right/match-true": 1}
events, event_ids = mne.events_from_annotations(eeg_raw, event_id=id_dict, verbose=False)

# The first column contains the event onset (in samples) with first_samp included. The last column contains the event
# code. The second column contains the signal value of the immediately preceding sample, and reflects the fact that
# event arrays sometimes originate from analog voltage channels (“trigger channels” or “stim channels”). In most cases,
# the second column is all zeros and can be ignored.

tmin, tmax = -2, 2  # in s
baseline = None
eeg_epochs = mne.Epochs(
    eeg_raw,
    events=events,
    event_id=event_ids,
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    verbose=False)

## plot epochs
eeg_epochs.plot(n_epochs=10, picks=['eeg'])

## visualize mean epochs (mean over channels)
eeg_epochs.plot_image(combine="mean")

## import the sklearn library
from sklearn.model_selection import train_test_split
from sklearn import svm

## extract array data from the MNE object
n_events = events.shape[0]
X = eeg_epochs.get_data(mne.pick_channels(eeg_raw.info['ch_names'],
                                          include=eeg_raw.info['ch_names'],
                                          exclude=eeg_raw.info['bads'])).reshape(n_events, -1)

## assign events to the labels variable y
y = events[:, -1]

## confirm that the lengths of inputs and labels are the same
assert X.shape[0] == y.shape[0], 'inputs and labels do not have the same length'

## split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=98)

## train the classifier
clf = svm.SVC(kernel="linear", C=1)
clf.fit(x_train, y_train)
print(f'train accuracy: {clf.score(x_train, y_train)}')
print(f'test accuracy: {clf.score(x_test, y_test)}')

