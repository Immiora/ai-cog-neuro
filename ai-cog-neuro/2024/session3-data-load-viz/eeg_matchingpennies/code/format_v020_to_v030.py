"""Reformat eeg_matchingpennies from version 0.2.0 to version 0.3.0.

In the changes from 0.1.6 to 0.2.0, two additional events per trial were
added as rows to the events.tsv files, corresponding to the onset and offset
of a "shrinking bar" countdown that was part of the experimental procedure.
Adding these events turned the previous "by-trial" organization of the
events.tsv file to a mix of "by-trial" (i.e., one row per trial) and
"by-event" (i.e., one row per event) organization, which is inconsistent
and can be confusing for users.
To format the dataset into a full "by-event" format, which is often preferable,
too little information was saved during data recording (not all events can
be accurately reconstructed).
So with this present reformatting script, the dataset events files are turned
back into a pure "by-trial" format.

To further document the data, Hierarchical Event Descriptor (HED) tags are
added.

Requirements
------------
- Python 3.7.7
- Pandas 1.0.5

"""
# %% Imports
import os.path as op
import re
from datetime import datetime
import json

import pandas as pd

# %% Constants
mp_root = "/home/stefanappelhoff/Desktop/bids/bids-examples/eeg_matchingpennies"

countdown_duration_s = 3.0

# %% Assert we are working on the correct version of the data

CHANGES = op.join(mp_root, "CHANGES")
with open(CHANGES, "r", encoding="utf-8") as fin:
    lines = fin.readlines()

expected_versions = set(
    ["0.1.0", "0.1.1", "0.1.2", "0.1.3", "0.1.4", "0.1.5", "0.1.6", "0.2.0"]
)
found_versions = []
for line in lines:
    match = re.match(r"(\d\.\d\.\d)", line)
    if match:
        found_versions.append(match.group(0))

diff = set(found_versions) - set(expected_versions)
if len(diff) > 0:
    raise RuntimeError(f"Found unexpected versions in CHANGES: {diff}")

# %% Collect changes that are done to data in a list

changes = []

# %% Update README

readme_txt = """Overview
--------
This is the "Matching Pennies" dataset. It was collected as part of a small
scale replication project targeting the following reference:

Matthias Schultze-Kraft et al.
"Predicting Motor Intentions with Closed-Loop Brain-Computer Interfaces".
In: Springer Briefs in Electrical and Computer Engineering.
Springer International Publishing, 2017, pp. 79–90.

In brief, it contains EEG data for 7 subjects raising either their left or right hand,
thus giving rise to a lateralized readiness potential as measured with the EEG.
For details, see the `Details about the experiment` section.


Citing this dataset
-------------------
Please cite as follows:

Appelhoff, S., Sauer, D., & Gill, S. S. (2018, July 22).
Matching Pennies: A Brain Computer Interface Implementation Dataset.

For more information, see the `dataset_description.json` file.


License
-------
This eeg_matchingpennies dataset is made available under the Open Database
License: See the LICENSE file. A human readable information can be found at:

https://opendatacommons.org/licenses/odbl/summary/

Any rights in individual contents of the database are licensed under the
Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/


Format
------
The dataset is formatted according to the Brain Imaging Data Structure. See the
`dataset_description.json` file for the specific version used.

Generally, you can find data in the .tsv files and descriptions in the
accompanying .json files.

An important BIDS definition to consider is the "Inheritance Principle", which
is described in the BIDS specification under the following link:

https://bids-specification.rtfd.io/en/stable/02-common-principles.html#the-inheritance-principle

The section states that:

> Any metadata file (such as .json, .bvec or .tsv) may be defined at any directory level,
> but no more than one applicable file may be defined at a given level [...]
> The values from the top level are inherited by all lower levels unless
> they are overridden by a file at the lower level.


Details about the experiment
----------------------------
For a detailed description of the task, see Schultze-Kraft et al. (2017)
and the supplied `task-matchingpennies_eeg.json` file.
What follows is a brief summary.

Participants were seated in front of a computer screen placed on a desk.
In front of the participants, two computer mice were glued to the desk:
One to the right of the current participant, the other one to their left.
On each trial in the experiment, participants were instructed to rest
their hands on the desk in front of them, using their left and
right ring-fingers to press a button on the computer mouse on their
left and right side respectively.
Pressing both mice buttons triggered a 3 second countdown that was
visually represented as a shrinking horizontal bar on the computer screen.
Participants were instructed to lift one (and only one!) hand, thereby
releasing a button of a computer mouse, at exactly the time when the
countdown reached 0 seconds.
After a short delay (see below), a feedback stimulus in the form of a
left or right hand was shown on the respective side of the computer screen.
When the hand shown on the computer screen and the hand raised by
the participant matched, this was counted as a "win" for the computer,
else it was counted as a "win" for the participant.
Participants were instructed to win as many trials against the computer
as possible.

Throughout the experiment, EEG data was recorded and analyzed in a closed-loop
BCI online setup. The computer made use of this data to predict which
hand a given study participant was about to raise, in order to win more trials.

For more information, you can also consult the events.tsv and events.json files.

Due to an inherent latency of the system, the data used in
the online analysis does not necessarily coincide with the offline data marked
through EEG triggers. Therefore, after each trial-wise online analysis, the
used data was saved separately as an "online chunk". Based on the online chunks
of data, it was possible to calculate a latency of each trial with respect to
the time difference between when an event happened at the electrode level and
when it arrived in digitized format at the classification function. These
latencies were calculated by sliding the online chunk of data across the data
recording until a perfect match was found. The index of this perfect match is
then compared to the timepoint of the event trigger and that difference
constitutes the latency. The values are included in the events.tsv files for
each subject.

The original data was recorded in `.xdf` format using labstreaminglayer
(https://github.com/sccn/labstreaminglayer). It is stored in the `/sourcedata`
directory. To comply with the BIDS format, the .xdf format was converted to
BrainVision format (see the `.eeg` file for binary eeg data, the `.vhdr` as a
text header filer containing meta data, and the `.vmrk` as a text file storing
the eeg markers).

Note that in the experiment, the only EEG TTL triggers that were recorded are
related to the participant raising their left (=value 1) or right (=value 2) hand,
and the button release associated with this movement.
The onset of the feedback stimulus (drawing of left or right hand shown on the screen)
was not recorded, however it can be crudely estimated to be around 50ms + ~10ms + ~8ms,
where 50ms were a hardcoded break before getting the online data from LSL, ~10ms was
the time to get a prediction from the BCI classifier, but this varied around 10ms,
and finally the ~8ms at the end come from the uncertainty of the refresh rate of the
screen (60fps = 1 screen every 16.66666ms)

Subjects 1 to 4 participated in the pilot testing only. Their data are not
included in this dataset.

"""

README = op.join(mp_root, "README")
with open(README, "w", encoding="utf-8") as fout:
    print(readme_txt, file=fout)

changes.append('Fixed typos in README file, and updated "inheritance principle" quote')


# %% Update CHANGES
changes_txt = """0.2.0 2020-07-24
    - Added info on feedback onset to README
    - Re-calculated onset column without rounding
    - Work on events.tsv and json
        - Renamed columns: trl->trial, pred->bci_prediction, pred_valid->bci_prediction_valid, n_invalid->n_repeated
        - Extend/clarify several 'Description' fields
        - Add previously undocumented events based on what was learned from experiment code
    - Use CMIXF-12 formatting throughout: https://people.csail.mit.edu/jaffer/MIXF/CMIXF-12
    - Add new recommended DatasetType field to dataset_description

0.1.6 2020-06-02
    - bumped BIDS version, anticipating the imminent release of BIDS 1.4
    - removed the unneeded .bidsignore file: LICENSE files are now valid, no need to ignore anymore
    - adjusted HowToAcknowledge field in dataset_description

0.1.5 2019-05-04
    - added the full license text in LICENSE and a .bidsignore file to make
      the bids-validator accept this file
    - Make adjustments to the License part in the README
    - adjusted the BIDS version used to 1.2, because the EEG extension proposal
      has been merged fully by now

0.1.4 2018-11-29
    - replaced uV strings with µV, using the micro sign µ
    - renamed event_value to value, and event_sample to sample
    - reformat README to have line lengths < 80 characters
    - remove superfluous "LongName" keys from JSON files
    - general BEP_006 compliance
    - remove subject wise eeg.json sidecar files, these are covered
      by the eeg.json at the root via inheritance

0.1.3 2018-09-21
    - added participants.tsv to explain additional variables: age,
      handedness, sex

0.1.2 2018-07-22
    - added DOI to dataset
    - changed events.tsv structure now using stim_file
    - changed events.tsv structure now using trial_type instead
      of lift_side
    - changed events.tsv structure: trial_type and pred now use left
      and right instead of 1 and 2
    - adjusted events.json according to events.tsv
    - general BEP_006 compliance

0.1.1 2018-06-13
    - adjusted BIDS formatting in compliance with BEP_006 EEG

0.1.0 2018-05-03
    - initial release
"""

CHANGES = op.join(mp_root, "CHANGES")
with open(CHANGES, "w", encoding="utf-8") as fout:
    print(changes_txt, file=fout)

changes.append(
    "Changed the order of changelog items: More recent items are now at the top"
)

# %% Drop "bci_prediction_valid" column, and countdown rows

# First assert that all values there are "1" (i.e., "valid").
# If they are, there is no point in having this column.
all_ones_list = []
fnames_events_tsv = []
for sub in range(5, 12):
    fname = op.join(
        mp_root,
        f"sub-{sub:02}",
        "eeg",
        f"sub-{sub:02}_task-matchingpennies_events.tsv",
    )
    df = pd.read_csv(fname, sep="\t")
    if df["bci_prediction_valid"].nunique() == 1:
        all_ones_list.append(True)
    else:
        all_ones_list.append(False)
    fnames_events_tsv.append(fname)

assert all(all_ones_list)
for fname in fnames_events_tsv:
    df = pd.read_csv(fname, sep="\t")

    # drop column
    df = df.drop(columns="bci_prediction_valid")

    # drop the rows
    df = df[~df["duration"].isna()]

    # make sure duration is int "0"
    df["duration"] = df["duration"].astype(int)

    df.to_csv(fname, sep="\t", na_rep="n/a", index=False)

events_json = op.join(mp_root, "task-matchingpennies_events.json")
with open(events_json, "r") as fin:
    events_json_dict = json.load(fin)

events_json_dict.pop("bci_prediction_valid", None)

with open(events_json, "w") as fout:
    json.dump(events_json_dict, fout, ensure_ascii=False, indent=4)
    fout.write("\n")

# add this change to CHANGES
changes += [
    "Dropped the 'bci_prediction_valid' column from events.tsv and its entry in events.json",
    "Dropped the event rows associated with countdown start, and countdown end from events.tsv",
]

# %% Add hand_raised, countdown on/offset and feedback onset, and reorganize df

for fname in fnames_events_tsv:

    # load data
    df = pd.read_csv(fname, sep="\t")

    df["hand_raised"] = df["value"].map({1: "left", 2: "right"})

    # add columns
    df["countdown_offset"] = df["onset"] - df["response_time"] / 1000
    df["countdown_onset"] = df["countdown_offset"] - countdown_duration_s

    # See README for justification of these values
    deliberate_delay = 0.05
    computation_delay_approx = 0.01
    fps = 60
    screen_refresh_delay_approx = (1 / fps) / 2
    df["feedback_onset_approx"] = (
        df["onset"]
        + deliberate_delay
        + computation_delay_approx
        + screen_refresh_delay_approx
    )

    # reformat levels of trial_type
    def comb(df):
        """Combine two string columns into one."""
        series = f"raised-{df['hand_raised']}/match-{df['hand_raised']==df['bci_prediction']}"
        return series.lower()

    df["trial_type"] = df.apply(comb, axis=1)

    # reorganize DF
    event_cols = [
        "onset",
        "duration",
        "trial",
        "hand_raised",
        "value",
        "sample",
        "countdown_onset",
        "countdown_offset",
        "response_time",
        "feedback_onset_approx",
        "stim_file",
        "trial_type",
        "stage",
        "bci_prediction",
        "latency",
        "n_repeated",
    ]
    df = df[event_cols]

    df.to_csv(fname, sep="\t", na_rep="n/a", index=False)

# add this change to CHANGES
changes += [
    "Added events.tsv columns hand_raised, countdown_onset, countdown_offset, and feedback_onset_approx",
    "Reformatted values of trial_type column",
    "Reorganized the column order in events.tsv",
]

# %% Update documentation in events.json file and add HED tags

events_json = op.join(mp_root, "task-matchingpennies_events.json")
with open(events_json, "r") as fin:
    events_json_dict_old = json.load(fin)

# correct order
events_json_dict = {
    key: events_json_dict_old.get(key, {"Description": ""}) for key in event_cols
}

# update/improve doc
events_json_dict["onset"][
    "Description"
] = "Onset of a button release (either left or right), corresponding to raising of a hand."
events_json_dict["duration"][
    "Description"
] = "Duration of the button released. Modeled as instantaneous event, so the duration is 0 seconds."
events_json_dict["trial"][
    "Description"
] = "The current trial within this stage (see 'stage')."
events_json_dict["hand_raised"][
    "Description"
] = "Which hand was raised (i.e., which button was released)."
events_json_dict["hand_raised"]["Levels"] = {
    "left": "The left hand was raised (i.e., left button was released)",
    "right": "The right hand was raised (i.e., right button was released)",
}
events_json_dict["sample"][
    "Description"
] = "The sample within the EEG data at which the button release event occurred (i.e., a hand was raised)."

events_json_dict["countdown_onset"][
    "Description"
] = "Onset of the countdown after which to raise a hand (duration: 3 s)."
events_json_dict["countdown_onset"]["Units"] = "s"

events_json_dict["countdown_offset"][
    "Description"
] = "Offset of the countdown. Must raise a hand now."
events_json_dict["countdown_offset"]["Units"] = "s"

events_json_dict["feedback_onset_approx"][
    "Description"
] = "Approximate onset of the bci-produced feedback (either a left or a right hand visual stimulus)."
events_json_dict["feedback_onset_approx"]["Units"] = "s"

events_json_dict["stim_file"][
    "Description"
] = "Name of the stimulus file that was shown during this trial at 'feedback_onset_approx'."

events_json_dict["trial_type"][
    "Description"
] = "Which hand was raised, and did this match the BCIs prediction (lose outcome), or not (win outcome)."
events_json_dict["trial_type"]["Levels"] = {
    "raised-left/match-false": "raised left hand, bci predicted right hand",
    "raised-left/match-true": "raised left hand, bci predicted left hand",
    "raised-right/match-false": "raised right hand, bci predicted left hand",
    "raised-right/match-true": "raised right hand, bci predicted right hand",
}
events_json_dict["trial_type"]["HED"] = {
    "raised-left/match-true": (
        "(Agent-action, Participant-response, (Lift, (Hand, (Left-side-of, Experiment-participant))), (Release, (Mouse-button, (Left-side-of, Experiment-participant)))), "
        "(Delay, Duration, (Approximately-equal-to, Time-block/68 ms)), "
        "(Sensory-event, Visual-presentation, (Drawing, (ID/left_hand.png, Hand, (Left-side-of, Computer-screen))), (Feedback, Penalty)), "
        "Description/Subject raised left hand and computer presented image of hand to left side"
    ),
    "raised-left/match-false": (
        "(Agent-action, Participant-response, (Lift, (Hand, (Left-side-of, Experiment-participant))), (Release, (Mouse-button, (Left-side-of, Experiment-participant)))), "
        "(Delay, Duration, (Approximately-equal-to, Time-block/68 ms)), "
        "(Sensory-event, Visual-presentation, (Drawing, (ID/right_hand.png, Hand, (Right-side-of, Computer-screen))), (Feedback, Reward)), "
        "Description/Subject raised left hand and computer presented image of hand to right side"
    ),
    "raised-right/match-true": (
        "(Agent-action, Participant-response, (Lift, (Hand, (Right-side-of, Experiment-participant))), (Release, (Mouse-button, (Right-side-of, Experiment-participant)))), "
        "(Delay, Duration, (Approximately-equal-to, Time-block/68 ms)), "
        "(Sensory-event, Visual-presentation, (Drawing, (ID/right_hand.png, Hand, (Right-side-of, Computer-screen))), (Feedback, Penalty)), "
        "Description/Subject raised right hand and computer presented image of hand to right side"
    ),
    "raised-right/match-false": (
        "(Agent-action, Participant-response, (Lift, (Hand, (Right-side-of, Experiment-participant))), (Release, (Mouse-button, (Right-side-of, Experiment-participant)))), "
        "(Delay, Duration, (Approximately-equal-to, Time-block/68 ms)), "
        "(Sensory-event, Visual-presentation, (Drawing, (ID/left_hand.png, Hand, (Left-side-of, Computer-screen))), (Feedback, Reward)), "
        "Description/Subject raised right hand and computer presented image of hand to left side"
    ),
}

events_json_dict["stage"][
    "Description"
] = "The current 'stage' of the experiment, which relates to how the BCI makes predictions."
events_json_dict["stage"]["Levels"] = {
    "1": "Stage 1. BCI makes random predictions.",
    "2": "Stage 2. BCI makes predictions after being trained on data from stage 1.",
    "3": "Stage 3. BCI makes predictions after being trained on data from stages 1 and 2.",
}

events_json_dict["bci_prediction"][
    "Description"
] = "What hand (left/right) did the BCI predict to be raised."

events_json_dict["latency"]["Description"] = (
    "Estimated beginning of the data chunk used for the BCI prediction with respect to 'onset' "
    "(when a hand was raised / a button was released)."
)

events_json_dict["n_repeated"]["Description"] = (
    "In case the participant did not adhere to the task rules, "
    "the trial was aborted and repeated (e.g., if a hand was raised before the countdown was over). "
    "'n_repeated' tracks how many trials in total within this 'stage' "
    "had to be repeated before the trial described in a given row."
)


# write back to file
with open(events_json, "w") as fout:
    json.dump(events_json_dict, fout, ensure_ascii=False, indent=4)
    fout.write("\n")

changes += [
    "Updated events.json, and improved the documentation therein",
    "Added HED tags to events.json file",
]

# %% Add HEDVersion field to dataset_description.json, bump BIDSVersion

dataset_description_json = op.join(mp_root, "dataset_description.json")

with open(dataset_description_json, "r") as fin:
    dataset_description = json.load(fin)

dataset_description["BIDSVersion"] = "1.6.0"
dataset_description["HEDVersion"] = "8.0.0"

with open(dataset_description_json, "w") as fout:
    json.dump(dataset_description, fout, ensure_ascii=False, indent=4)
    fout.write("\n")

changes += [
     "Added HEDVersion field to dataset_description.json file",
     "Bumped BIDSVersion field in dataset_description.json to 1.6.0",
]

# %% Update contents of CHANGES

update_txt = f"0.3.0 {datetime.now().strftime('%Y-%m-%d')}"
for change in changes:
    update_txt += "\n    - " + change

CHANGES = op.join(mp_root, "CHANGES")
with open(CHANGES, "r", encoding="utf-8") as fin:
    lines = fin.read()

update_txt += "\n\n" + lines

with open(CHANGES, "w", encoding="utf-8") as fout:
    print(update_txt.rstrip(), file=fout)
