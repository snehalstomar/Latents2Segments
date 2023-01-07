# Latents2Segments: Disentangling the Latent Space of Generative Models for Semantic Segmentation of Face Images
CVPR Workshop on Computer Vision for Augmented and Virtual Reality (CV4ARVR), New Orleans, Louisiana 2022

> + Authors: [Snehal Singh Tomar](https://www.snehalstomar.github.io) and [A.N. Rajagopalan](https://www.ee.iitm.ac.in/raju/)
> + Paper: [CVPRW 2022 arXiv preprint](https://arxiv.org/abs/2207.01871)

<p align="center">
  <img src="assets/cv4arvr.gif" alt="Semantic Segmentation: Qualitative Results" width="600" />
</p>


## Setup

Our setup for this project entailed the following:

> + CUDA 10.0, cuDNN 7.5.0, Python 3.6, Pytorch 1.7.1, and Ubuntu 20.04.
> + Python packages: dominate torchgeometry func-timeout tqdm matplotlib opencv_python lmdb numpy GPUtil Pillow scikit-learn visdom ninja
> + Upon cloning the repository, please place the  ROI-separated [CelebAMask-HQ Dataset](https://github.com/switchablenorms/CelebAMask-HQ) and place it within the cloned directory, in the following directory structure before running any experiments:
```
swapping_style_controlled_AE_dataset/
├── test
│	├── eyes
│	├── full
│	├── hair
│	├── lips
│	├── nose
│	└── skin
└── train
    ├── eyes
    ├── full
    ├── hair
    ├── lips
    ├── nose
    └── skin
```

## Training

Please run:

> python train.py --dataroot swapping_style_controlled_AE_dataset/train --dataset_mode imagefolder --checkpoints_dir checkpoints --num_gpus 1 --batch_size 2 --preprocess resize --load_size 128 --crop_size 128 --name <"desired_model_name"> --evaluation_metrics swap_visualization --evaluation_freq 100 --save_freq 3000 --continue_train True

The trained model is saved at "checkpoints/desired_model_name". 

## Inference

To generate predicted segmentation maps, run:

> python s_s_explorer.py --evaluation_metrics simple_swapping --preprocess scale_shortside --load_size 128 --crop_size 128 --checkpoints_dir <"path_to_weight_directory"> --name <"trained_model_name"> --input_structure_image swapping_style_controlled_AE_dataset/test/full/filename.png --input_texture_image swapping_style_controlled_AE_dataset/test/filename.png --dataroot swapping_style_controlled_AE_dataset/test

### Bibtex
If you use this code, please cite our paper:	
```
@inproceedings{tomar2022Lat2seg,
  title={Latents2Segments: Disentangling the Latent Space of Generative Models for Semantic Segmentation of Face Images},
  author={Tomar, Snehal Singh and Rajagopalan, A.N.},
  booktitle={CVPR Workshop on Computer Vision for Augmented and Virtual Reality (CV4ARVR), New Orleans, Louisiana},
  year={2022}
}
```

### License

This code is for non-commercial use only. Please refer to our License file for more.

### Acknowledgement

This implementation borrows substantially from the [Swapping Autoencoder](https://github.com/taesungp/swapping-autoencoder-pytorch). 