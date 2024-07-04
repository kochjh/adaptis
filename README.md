## Implementing AdaptIS on custom dataset
This is a fork of the AdaptIS repository. AdaptIS framework is used for a custom dataset. The original README can be found below, first the changes and additions made are described.

### Custom dataset
The dataset consists of 5 classes (+ background class) and 630 high-resolution images (447 train, 93 validation, 90 test). The images have a resolution 1536 x 1536 pixel. For each image a corresponding mask is given. The intensity of each pixel in the mask image determines the class ID and instance ID of the pixel. Dividing the intensity by 4 results in a two-digit number, where the left digit is the class ID and the right digit is the instance ID.

### Implemented changes
Since AdaptIS implements panoptic segmentation for a single class but the custom dataset consists of 5 classes, changes were necessary. All changes are listed below:

* Minor changes due to depracted functions in some used packages
* A new dataset (`adaptis/data/robotec.py`) has been created. The dataset inherits from the existing class and is responsible for loading the images and decoding the masks accordingly. Since every instance in AdaptIS is assigned to one class ID, I have defined the combination of class and instance as an instance in AdaptIS. If this is not the desired behavior, `instance_mask` can be used instead of `mask` in `robotec.py` line 49. 
* A new model (`adaptis/model/robotec`) has been created. The new model uses 6 output channels in the segmentation head. One channel per class (5 classes + background)
* A training script (`train_robotec.py`) has been created. The training script is also based on the existing training scripts and defines some hyperparameters. One difficulty was adapting the training to the given hardware. To achieve this, the images were downsampled and the number of proposal points reduced. Some parameters are shown in the following table, all other hyperparameters are the same as for the original AdaptIS training.

training without proposal-head:
| Parameter | Value |
|-----------|------|
|   image size  |    0.6 * 1536 = 922  |
|   epochs   |   60   |
|   proposal points | 6 |
| batch size | 1 |

following training with proposal-head:
| Parameter | Value |
|-----------|------|
|   image size  |    0.6 * 1536 = 922  |
|   epochs   |   10  |
|   proposal points | 20 |
| batch size | 1 |

### Alternative approach
I also thought about using a pretrained model as backbone instead of the unet. To realize this, the input data must be transformed according to the pretrained model and the pretrained model must be integrated into the given architecture. I tried this (see the `pretrained` branch), but could not train the model properly due to my limited GPU memory. (All changes summarized in this [commit](https://github.com/kochjh/adaptis/commit/f3cfc310b74b9f536265c362ff70aad9982177f2))

### Usage
All my changes can be used like the original repository (see below). I made some small changes to the `requirements.txt` for pytorch CUDA support.

### Results

The results are visualized in `notebooks\test_robotec.ipynb`. I included a visualization of the semantic segmentation and the semantic segmentation ground truth.

Some comments / observations:
* I have trained with different resolutions. I noticed that the higher the resolution, the better the results. However, due to the hardware available to me, I could not use an even higher resolution
* I trained for 60/10 epochs. Longer training could change the results
* The semantic segmentation can segment well between background and non-background. However, the exact determination of the class is not reliably possible from this model. As some of the classes are very similar, a larger amount of training data may be necessary here



the original AdaptIS readme follows

--------------------------------


## AdaptIS: Adaptive Instance Selection Network
This codebase implements the system described in the paper ["AdaptIS: Adaptive Instance Selection Network"](https://arxiv.org/abs/1909.07829), Konstantin Sofiiuk, Olga Barinova, Anton Konushin. Accepted at ICCV 2019.
The code performs **panoptic segmentation** and can be also used for **instance segmentation**.

<p align="center">
  <img src="./images/adaptis_model_scheme.png" alt="drawing" width="600"/>
</p>


### ToyV2 dataset
![alt text](./images/toy2_wide.jpg)

We generated an even more complex synthetic dataset to show the main advantage of our algorithm over other detection-based instance segmentation algorithms. The new dataset contains 25000 images for training and 1000 images each for validation and testing. Each image has resolution of 128x128 and can contain from 12 to 52 highly overlapping objects.

* You can download the ToyV2 dataset from [here](https://drive.google.com/open?id=1iUMuWZUA4wzBC3ka01jkUM5hNqU3rV_U). 
* You can test and visualize the model trained on this dataset using [this](notebooks/test_toy_v2_model.ipynb) notebook.
* You can download pretrained model from [here](https://drive.google.com/open?id=1fq72ZeVdOHM37Qv648lRVVD0VWjcD_a2).

![alt text](./images/toy_v2_comparison.jpg)


### ToyV1 dataset

We used the ToyV1 dataset for our experiments in the paper. We generated 12k samples for the toy dataset (10k for training and 2k for testing). The dataset has two versions:
* **original** contains generated samples without augmentations;
* **augmented** contains generated samples with fixed augmentations (random noise and blur).

We trained our model on the original/train part with online augmentations and tested it on the augmented/test part. The repository provides an example of testing and metric evalutation for the toy dataset.
* You can download the toy dataset from [here](https://drive.google.com/open?id=161UZrYSE_B3W3hIvs1FaXFvoFaZae4FT). 
* You can test and visualize trained model on the toy dataset using [provided](notebooks/test_toy_model.ipynb) Jupyter Notebook.
* You can download pretrained model from [here](https://drive.google.com/file/d/1n1UzzNN_9H2F71xyhKckJDr8XHDSJ-py).


### Setting up a development environment

AdaptIS is built using Python 3.6 and relies on the most recent version of PyTorch. This code was tested with PyTorch 1.3.0 and TorchVision 0.4.1. The following command installs all necessary packages:

```
pip3 install -r requirements.txt
```

Some of the inference code is written using Cython, you must compile the code before testing:
```
make -C ./adaptis/inference/cython_utils
```


### Training

Currently our implementation supports training only on single gpu, which can be selected through *gpus* flag.

You can train model for the ToyV2 dataset by the following command:
```
python3 train_toy_v2.py --batch-size=14 --workers=2 --gpus=0 --dataset-path=<toy-dataset-path>
```

You can train model for the toy dataset (original from the paper) by the following command:
```
python3 train_toy.py --batch-size=14 --workers=2 --gpus=0 --dataset-path=<toy-dataset-path>
```


### License
The code of AdaptIS is released under the MPL 2.0 License. MPL is a copyleft license that is easy to comply with. You must make the source code for any of your changes available under MPL, but you can combine the MPL software with proprietary code, as long as you keep the MPL code in separate files.


### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/1909.07829).

```
@article{adaptis2019,
  title={AdaptIS: Adaptive Instance Selection Network},
  author={Konstantin Sofiiuk, Olga Barinova, Anton Konushin},
  journal={arXiv preprint arXiv:1909.07829},
  year={2019}
}
```
