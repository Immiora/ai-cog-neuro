#  1. Gunzip (Nipype)
#  2. Motion Correction (FSL)

## Download data
import openneuro as on
bids_dir = 'data/bids'
subject = 'sub-07'
on.download(dataset='ds003688', target_dir=bids_dir, include=subject)

## Specify paths to the data
from nipype.interfaces.io import BIDSDataGrabber
bg = BIDSDataGrabber()
bg.inputs.base_dir = bids_dir
bg.inputs.subject = '07'
results = bg.run()

## Get the Node and Workflow object
from nipype import Node, Workflow

## Create a workflow
preproc = Workflow(name='work_preproc', base_dir='output/')

## 1. Gunzip (Nipype)
from nipype.algorithms.misc import Gunzip

# Specify example input file
func_file = results.outputs.bold[0]

# Initiate Gunzip node
gunzip_func = Node(Gunzip(in_file=func_file), name='gunzip_func')

##  2. Motion Correction (FSL)
from nipype.interfaces.fsl import MCFLIRT

mcflirt = Node(MCFLIRT(mean_vol=False,
                       save_plots=True),
                       name="mcflirt")

# Connect MCFLIRT node to the other nodes here
preproc.connect([(gunzip_func, mcflirt, [('out_file', 'in_file')])])

## Save the workflow graph
preproc.write_graph(graph2use='colored', format='png', simple_form=True)

## Run the workflow
preproc.run('MultiProc', plugin_args={'n_procs': 4})

## Load events
import pandas as pd
trialinfo = pd.read_table('data/bids/sub-07/ses-mri3t/func/sub-07_ses-mri3t_task-film_run-1_events.tsv')

## Load processed images
import nibabel as nib
Nifti_img  = nib.load('output/work_preproc/mcflirt/sub-07_ses-mri3t_task-film_run-1_bold_mcf.nii.gz')
nii_data = Nifti_img.get_fdata() # 641 images of 64x64x40

## Split the data into cross-validation folds

## Train a classifier to predict labels: 1 for speech and 0 for music

## Try different classifiers and parameters