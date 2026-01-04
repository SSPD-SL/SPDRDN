# SPDRDN
Code
This repository implements a SPDRDN–based model for polarimetric image reconstruction.
Different polarization parameters (AoP, DoLP, DoCP, DoP) are processed independently using separate but identical RDN branches, rather than being mixed in a single network.
Each branch is trained to reconstruct one polarimetric component from noisy polarimetric images.

##################################################################
Usage
Train
1. cd AOP-branch
2. python train.py

Test
1. cd AOP-branch
2. python test.py

(Replace AOP-branch with DOLP-branch, DOCP-branch, or DOP-branch as needed.)
##################################################################

Calculate polarimetric parameters
Running the “Full_Stokes_parameters.m”
##################################################################

The “data” folder contains 50 sets of photon-counting images for specific state of polarization. Each “Norm_photon.mat” file stores four-channel intensity images captured under different scenes, arranged in channel order.
