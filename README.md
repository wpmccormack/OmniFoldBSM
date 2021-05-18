# OmniFoldBSM

This git repository contains the code used for the study detailed in <insert arxiv link when available>.  It relies on the OmniFold code found here: https://github.com/ericmetodiev/OmniFold.

<Talk about the Zenodo files>

In general, this study proceeds in three steps:
1. An npz file is made that contains pre-formatted numpy arrays for use in Keras NN training.
2. Unfolding is performed with the OmniFold method.
3. Plots are made with Jupyter notebooks.

## Repository contents

### npzMakers
* This folder contains python macros that were used to create the npz files used for training.  They are run using the sh files detailed in the next bullet.  Naming conventions (to understand what's in the macro.  Nothing relies on this particular format.):
  * Files start with omni_npz_maker
  * If "multifold" is in the name, then the file is designed to be used for multifold unfolding, rather than full particle-level unfolding
  * If "synthsig" is in the name, then synthetic signal has been included in the prior for the unfolding.  If "NOsynthsig" is in the name, then there is not synthetic signal in the prior
  * If "not125" is in the name, then the signal comes from a sample with the Higgs mass set to 250 GeV
  * If "synthsigOnly125" is in the name, then the synthetic signal in the prior only comes from the 125 GeV Higgs mass case
* The most recent macros (in chronological order) are:
  * omni_npz_maker_synthsig_herwig_withBSM.py (This didn't end up getting used)
  * omni_npz_maker_NOsynthsig_omnifold_pythia.py
  * omni_npz_maker_NOsynthsig_omnifold_herwig.py
  * omni_npz_maker_NOsynthsig_multifold.py
  * omni_npz_maker_NOsynthsig_multifold_herwig.py
* These macros take two inputs: the mass slice for the signal and the percent contamination.  In the most recent macros, the mass slice options are 0-5, where 0 corresponds to m_a = 500 MeV, 1 corresponds to m_a = 1 GeV, and so forth, with 5 corresponding to m_a = 16 GeV.  A minor point about the naming of the output files is that files with 0.1% contamination have "0Perc" in the name, 1% has "1Perc", and 10% has "10Perc".

### npzMakers_shFiles
* This folder contains sh files used to run the npz making files.  The naming conventions match what is described above.  Please see the contents of each file to see what specific mass values and signal contaminations are used.