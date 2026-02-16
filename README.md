# AlphaQubit Project

This is a project following the paper https://www.nature.com/articles/s41586-024-08148-8 by the Google Deempind and Google Quantum AI teams. The paper looks at using a transformer-based RNN to decode error syndromes for different surface codes on their Sycamore chip. The idea is that although algorithms such as MWPM decoding might be theoretically optimal for a given noise model, realistic noise is actually much more complicated and hardware dependent, allowing machine learning models to outperform these algorithms. 

I was mainly doing this project just for learning purposes, both ML and QEC. I only look at the d=3 rotated z memory surface code and look at data from one specific location (3,5) on the Sycamore processor, for the sake of simplicity. I play around with a few ideas on making inference faster. In the paper, although the algorithm has SOTA accuracy for specific datasets, inference isn't fast enough for real time error correction, especially as we go to larger code distances (which will be needed in order to achieve fault tolerance). I play around with a few standard inference speed up mechanisms such as MoE (Mixture of Experts). I would also like to explore an interesting adaption of MoE, mentioned in the paper https://arxiv.org/abs/2509.01322, where some of the experts carry out no computation, depending on the difficulty of the input (syndrome), saving time and compute on inference.

# Data

This project uses the Google Sycamore surface code dataset.

I generate two different types of data for pre-training. I chose a mode to train the model on.
    - SI1000 data, is generated using stim and is meant to emulate generic noise on a superconducting chip, I apply a soft noise model to this data.
    - p_ij DEM data, this uses a DEM (a hypergraph where edges represent probabilities) which is derived from the experimental data.

I then fine-tune the model after pre-training on experimental data.

## Download for DEM data
1. Download from https://zenodo.org/records/6804040
2. Unzip google_qec3v5_experiment_data.zip

## Expected Structure
For d=3 Z-basis data you need the following folders:
    z_centre_3_5_d3_data/
        surface_code_bZ_d3_r01_center_3_5/
        surface_code_bZ_d3_r03_center_3_5/
        ...
        surface_code_bZ_d3_r25_center_3_5/

## Config
Set your data path in config_local.py:
    data_dir = "/path/to/z_centre_3_5_d3_data"