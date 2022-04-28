
<div align="center">
  <img src="assets/opensphere_logo2.png" width="600"/>
</div>
&nbsp;

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-Pytorch-brightgreen)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpenSphere** is a hyperspherical face recognition library based on PyTorch. Check out the [project homepage](https://opensphere.world/).

<p align="center"> 
<img src="assets/teaser.gif" width="580"/>
</p>

## Introduction
**OpenSphere** provides a consistent and unified training and evaluation framework for hyperspherical face recognition research. The framework decouples the loss function from the other varying components such as network architecture, optimizer, and data augmentation. It can fairly compare different loss functions in hyperspherical face recognition on popular benchmarks, serving as a transparent platform to reproduce published results.


<!-- TABLE OF CONTENTS -->
***Table of Contents***: - <a href="#key-features">Key features</a> - <a href="#setup">Setup</a> - <a href="#get-started">Get started</a> - <a href="#log-and-pretrained-models">Pretrained models</a> - <a href="#reproduce-published-results">Reproducible results</a> - <a href="#citation">Citation</a> -

<details open>
<summary>Supported Projects</summary>
	

- [x] [**SphereFace**: Deep Hypersphere Embedding for Face Recognition](https://wyliu.com/papers/LiuCVPR17v3.pdf), *CVPR 2017*</li>

- [x] [**SphereFace2**: Binary Classification is All You Need for Deep Face Recognition](https://wyliu.com/papers/sphereface2_ICLR22.pdf), *ICLR 2022* 

- [x] [**SphereFace Revived**: Unifying Hyperspherical Face Recognition](https://wyliu.com/papers/spherefacer_v3_TPAMI.pdf), *TPAMI 2022*

</details>


## Update
- **2022.4.1**: initial commit.
- **2022.4.9**: added the download script for some datasets.
- **2022.4.12**: added SFNet (with BN) and IResNet.
- **2022.4.22**: added SphereFace+, MS1M, and config files on MS1M.
- **2022.4.28**: added SphereFace-R, Glint360K, and more pretrained models.

## Key Features
- **Implemented Loss Functions** ([folder](https://github.com/ydwen/opensphere/tree/main/model/head))
  - [SphereFace](https://wyliu.com/papers/LiuCVPR17v3.pdf), [SphereFace+](https://wyliu.com/papers/LiuNIPS18_MHE.pdf), [SphereFace2](https://wyliu.com/papers/sphereface2_ICLR22.pdf), [SphereFace-R](https://wyliu.com/papers/spherefacer_v3_TPAMI.pdf)
  - [ArcFace](https://arxiv.org/pdf/1801.07698.pdf), [AM-Softmax](https://arxiv.org/pdf/1801.05599.pdf) ([CosFace](https://arxiv.org/pdf/1801.09414.pdf)), [CocoLoss](https://arxiv.org/pdf/1710.00870.pdf) ([NormFace](https://arxiv.org/pdf/1704.06369.pdf))

- **Implemented Network Architectures** ([folder](https://github.com/ydwen/opensphere/tree/main/model/backbone))
  - [SFNet (without Batch Norm)](https://wyliu.com/papers/LiuCVPR17v3.pdf), [SFNet (with Batch Norm)](https://wyliu.com/papers/spherefacer_v3_TPAMI.pdf), [IResNet](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

- **Available Datasets**
  - **Training**: [VGGFace2](https://arxiv.org/pdf/1710.08092.pdf), [WebFace](https://arxiv.org/pdf/1411.7923v1.pdf), [MS1M](https://arxiv.org/pdf/1607.08221.pdf), [Glint360K](https://arxiv.org/abs/2203.15565)
  - **Validation**: [LFW](http://vis-www.cs.umass.edu/lfw/), [AgeDB-30](https://ibug.doc.ic.ac.uk/media/uploads/documents/agedb.pdf), [CA-LFW](https://arxiv.org/pdf/1708.08197.pdf), [CP-LFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)
  - **Testing**: [IJB-B](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w6/papers/Whitelam_IARPA_Janus_Benchmark-B_CVPR_2017_paper.pdf), [IJB-C](http://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf)

- **TODO**: [SphereFace-R](https://wyliu.com/papers/spherefacer_v3_TPAMI.pdf) and more examplar config files

- We welcome submissions of your loss functions or network architectures to **OpenSphere**!


## Setup
1. Clone the OpenSphere repository. We'll call the directory that you cloned OpenSphere as `$OPENSPHERE_ROOT`.

    ```console
    git clone https://github.com/ydwen/opensphere.git
    ```

2. Construct virtual environment in [Anaconda](https://www.anaconda.com/):

    ```console
    conda env create -f environment.yml
    ```

## Get started
In this part, we assume you are in the directory `$OPENSPHERE_ROOT`. After successfully completing the [Setup](#setup), you are ready to run all the following experiments.

1. **Download and process the datasets**

  - Download the training set (`VGGFace2`), validation set (`LFW`, `Age-DB`, `CA-LFW`, `CP-LFW`), and test set (`IJB-B` and `IJB-C`) and place them in `data/train`, `data/val` amd `data/test`, respectively.
	
  - For convenience, we provide a script to automatically download the data. Simply run

	```console
	bash scripts/dataset_setup.sh
	```

  - If you need the `MS1M` training set, please run the additional commend:

	```console
	bash scripts/dataset_setup_ms1m.sh
	```
  - To download other datasets (e.g., `WebFace` or `Glint360K`), see the `scripts` folder and find what you need.

2. **Training a model (see the training config file for the detailed setup)**

    We give a few examples for training on different datasets with different backbone architectures:

  - To train SphereFace2 with SFNet-20 on `VGGFace2`, run the following commend (with 2 GPUs):

	```console
	CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/train/vggface2_sfnet20_sphereface2.yml
	```

  - To train SphereFace with SFNet-20 on `VGGFace2`, run the following commend (with 2 GPUs):

	```console
	CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/train/vggface2_sfnet20_sphereface.yml
	```

  - To train SphereFace with SFNet-64 (with BN) on `MS1M`, run the following commend (with 4 GPUs):

	```console
	CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config config/train/ms1m_sfnet64bn_sphereface.yml
	```

  - After finishing training a model, you will see a `project` folder under `$OPENSPHERE_ROOT`. The trained model is saved in the folder named by the job starting time, eg, `20220422_031705` for 03:17:05 on 2022-04-22.

  - Our framework also re-implements some other popular hyperspherical face recognition methods such as ArcFace, AM-Softmax (CosFace) and CocoLoss (NormFace). Please check out the folder `model/head` and some examplar config files in the folder `config/papers/SphereFace2/sec31`.

3. **Test a model (see the testing config file for detailed setup)**

  - To test on the `combined validation` dataset, simply run

	```console
	CUDA_VISIBLE_DEVICES=0,1 python test.py --config config/test/combined.yml --proj_dir project/##YourFolder##
	```

  - To test on both `IJB-B` and `IJB-C`, simply run

	```console
	CUDA_VISIBLE_DEVICES=0,1 python test.py --config config/test/ijb.yml --proj_dir project/##YourFolder##
	```

  - To test on `IJB-B`, simply run

	```console
	CUDA_VISIBLE_DEVICES=0,1 python test.py --config config/test/ijbb.yml --proj_dir project/##YourFolder##
	```

  - To test on `IJB-C`, simply run

	```console
	CUDA_VISIBLE_DEVICES=0,1 python test.py --config config/test/ijbc.yml --proj_dir project/##YourFolder##
	```

For more information about how to use training and testing config files, please see [here](https://github.com/ydwen/opensphere/tree/main/config).

## Results and pretrained models

<div align="center">
	
Loss | Architecture | Dataset | Config & Training Log & Pretrained Model
:---:|:---:|:---:|:---:
SphereFace | SFNet-20 (without BN) | VGGFace2 |[Google Drive](https://drive.google.com/file/d/1-NUP9cthANpa3_HHpEOklFtJC_dJNVjP/view?usp=sharing)
SphereFace+ | SFNet-20 (without BN) | VGGFace2 |[Google Drive](https://drive.google.com/file/d/1CBfxxTN712QmuwTk7i2az6JTs3ZRo-ia/view?usp=sharing)
SphereFace-R | SFNet-20 (without BN) | VGGFace2 | To be added
SphereFace2 | SFNet-20 (without BN) | VGGFace2 |[Google Drive](https://drive.google.com/file/d/1ZO3clpW_NHTybOgXIhrA7Kid4OIpQnrG/view?usp=sharing)
SphereFace | SFNet-64 (with BN) | MS1M |[Google Drive](https://drive.google.com/file/d/1UxQryEhy6UAlbg5rDJcybLemuvIzoMZ_/view?usp=sharing)
SphereFace+ | SFNet-64 (with BN) | MS1M | To be added
SphereFace-R | SFNet-64 (with BN) | MS1M | To be added
SphereFace2 | SFNet-64 (with BN) | MS1M | To be added
SphereFace | IResNet-100 | MS1M |[Google Drive](https://drive.google.com/file/d/1Te66jDL0uXdfnBxd5qTv_FvP5ES_EF5y/view?usp=sharing)
SphereFace+ | IResNet-100 | MS1M | To be added
SphereFace-R | IResNet-100 | MS1M | To be added
SphereFace2 | IResNet-100 | MS1M | To be added
SphereFace | SFNet-64 (with BN) | Glint360K | To be added
SphereFace+ | SFNet-64 (with BN) | Glint360K | To be added
SphereFace-R | SFNet-64 (with BN) | Glint360K | To be added
SphereFace2 | SFNet-64 (with BN) | Glint360K | To be added
SphereFace | IResNet-100 | Glint360K | To be added
SphereFace+ | IResNet-100 | Glint360K | To be added
SphereFace-R | IResNet-100 | Glint360K | To be added
SphereFace2 | IResNet-100 | Glint360K | To be added
</div>
	
## Reproduce published results

We create an additional folder `config/papers` that is used to provide detailed config files and reproduce results in published papers. Currently we provide config files for the following papers:
  
  - SphereFace2: Binary Classification is All You Need for Deep Face Recognition, ICLR 2022


## Citation

If you find **OpenSphere** useful in your research, please consider to cite:

For **SphereFace**:

  ```bibtex
  @article{Liu2022SphereFaceR,
	title={SphereFace Revived: Unifying Hyperspherical Face Recognition},
	author={Liu, Weiyang and Wen, Yandong and Raj, Bhiksha and Singh, Rita and Weller, Adrian},
	journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
	year={2022}
  }
	
  @InProceedings{Liu2017SphereFace,
	title = {SphereFace: Deep Hypersphere Embedding for Face Recognition},
	author = {Liu, Weiyang and Wen, Yandong and Yu, Zhiding and Li, Ming and Raj, Bhiksha and Song, Le},
	booktitle = {CVPR},
	year = {2017}
  }
  ```

For **SphereFace+**:

  ```bibtex
  @InProceedings{Liu2018MHE,
	title={Learning towards Minimum Hyperspherical Energy},
	author={Liu, Weiyang and Lin, Rongmei and Liu, Zhen and Liu, Lixin and Yu, Zhiding and Dai, Bo and Song, Le},
	booktitle={NeurIPS},
	year={2018}
  }
  ```

For **SphereFace2**:

  ```bibtex
  @InProceedings{wen2021sphereface2,
	title = {SphereFace2: Binary Classification is All You Need for Deep Face Recognition},
	author = {Wen, Yandong and Liu, Weiyang and Weller, Adrian and Raj, Bhiksha and Singh, Rita},
	booktitle = {ICLR},
	year = {2022}
  }
  ```
	
	

## Contact

  [Yandong Wen](https://ydwen.github.io) and [Weiyang Liu](https://wyliu.com)

  Questions can also be left as issues in the repository. We will be happy to answer them.
