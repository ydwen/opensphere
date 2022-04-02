<div align="center">
  <img src="assets/opensphere_logo.png" width="500"/>
</div>

## Introduction
OpenSphere is an open source face recognition toolbox based on PyTorch.


<details open>
<summary>Support Projects</summary>

- **SphereFace2: Binary Classification is All You Need for Deep Face Recognition**

- **SphereFace Revived: Unifying Hyperspherical Face Recognition** (todo)
  
</details>


## License
OpenSphere is released under the [MIT license](LICENSE).



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

## Usage
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
  
	@article{wen2021sphereface2,
	  title = {SphereFace2: Binary Classification is All You Need for Deep Face Recognition},
	  author = {Wen, Yandong and Liu, Weiyang and Weller, Adrian and Raj, Bhiksha and Singh, Rita},
	  booktitle = {arXiv preprint arXiv:2108.01513},
	  year = {2021}
	}

## Contact

  [Yandong Wen](https://ydwen.github.io) and [Weiyang Liu](https://wyliu.com)

  Questions can also be left as issues in the repository. We will be happy to answer them.
