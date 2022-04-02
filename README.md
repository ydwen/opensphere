
<div align="center">
  <img src="assets/opensphere_logo2.png" width="600"/>
</div>
&nbsp;

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-Pytorch-brightgreen)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpenSphere** is a hyperspherical face recognition library based on PyTorch. See the [project homepage](https://opensphere.world/).

## Introduction
**OpenSphere** provides a consistent and unified training and evaluation framework for hyperspherical face recognition research. The framework decouples the loss function from the other varying components such as network architecture, optimizer, and data augmentation. It can fairly compare different loss functions in hyperspherical face recognition on popular benchmarks, serving as a transparaent platform to reproduce published results.


<details open>
<summary>Supported Projects</summary>
	

- [x] **SphereFace**: Deep Hypersphere Embedding for Face Recognition, *CVPR 2017* </li>

- [x] **SphereFace2**: Binary Classification is All You Need for Deep Face Recognition, *ICLR 2022* </li>

- [ ] **SphereFace Revived**: Unifying Hyperspherical Face Recognition, *TPAMI 2022* </li>
  
</details>



## Update
- **2022.4.1**: initial commit.


## Setup (with [Anaconda](https://www.anaconda.com/))
1. Clone the OpenSphere repository. We'll call the directory that you cloned OpenSphere as **`OPENSPHERE_ROOT`**.

    ```Shell
    git clone https://github.com/ydwen/opensphere.git
    ```

2. Construct virtual environment in Anaconda

    ```Shell
    conda env create -f environment.yml
    ```

## Getting started
**Note:** In this part, we assume you are in the directory **`$OPENSPHERE_ROOT`**
*After successfully completing the [Setup](#setup)*, you are ready to run all the following experiments.

1. Datasets
Download the training set (`VGGFace2`), validation set (`LFW`, `Age-DB`, `CA-LFW`, `CP-LFW`), and test set (`IJB-B` and `IJB-C`) and place them in **`../data/`**.

	```Shell
	tar xvf ../data/vggface2.tar -C ../data/ 
	```
  
	```Shell
	tar xvf ../data/validation.tar -C ../data/ 
	```
  
	```Shell
	tar xvf ../data/IJB.tar -C ../data/
	```


2. Train

	```Shell
	CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/papers/SphereFace2/sec41/vggface2_sfnet20_sphereface2.yml
	```

3. Test
	```Shell
	CUDA_VISIBLE_DEVICES=0,1 python test.py --config config/test/ijbb.yml
	```


## Citation

If you find **OpenSphere** useful in your research, please consider to cite:

For **SphereFace**:

  	@article{Liu2022SphereFaceR,
	  title={SphereFace Revived: Unifying Hyperspherical Face Recognition},
	  author={Liu, Weiyang and Wen, Yandong and Raj, Bhiksha and Singh, Rita and Weller, Adrian},
	  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
	  year={2022}
	}
	
	@InProceedings{Liu2017CVPR,
	  title = {SphereFace: Deep Hypersphere Embedding for Face Recognition},
	  author = {Liu, Weiyang and Wen, Yandong and Yu, Zhiding and Li, Ming and Raj, Bhiksha and Song, Le},
	  booktitle = {CVPR},
	  year = {2017}
	}
	
      
For **SphereFace2**:
  
	@article{wen2021sphereface2,
	  title = {SphereFace2: Binary Classification is All You Need for Deep Face Recognition},
	  author = {Wen, Yandong and Liu, Weiyang and Weller, Adrian and Raj, Bhiksha and Singh, Rita},
	  booktitle = {arXiv preprint arXiv:2108.01513},
	  year = {2021}
	}
	
	

## Contact

  [Yandong Wen](https://ydwen.github.io) and [Weiyang Liu](https://wyliu.com)

  Questions can also be left as issues in the repository. We will be happy to answer them.
