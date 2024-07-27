# Orthogonal Annotation Cut Mix for Semi-Supervised Medical Image Segmentation (OACM)

## Introduction
In semi-supervised medical image segmentation tasks, complex 3D samples often suffer from inconsistent annotation quality, and annotations by different doctors can vary, failing to fully capture the 3D information of the samples. We propose an Orthogonal Annotation Cut Mix (OACM) method that integrates orthogonal annotation (OA) into the traditional Mean Teacher (MT) framework. By manually annotating two 2D slices and registering them to generate complete 3D coarse labels, we significantly reduce the complexity of manual annotation. Additionally, by guiding the cut mix between labeled and unlabeled samples through the Range of Interest (ROI), we better integrate information across different samples, enhancing model generalization and achieving better segmentation results without losing too much detail.

## Requirements
This project is developed with the following versions:

- Python 3.9
- PyTorch 2.0.0+cu118

Other dependencies and their versions:
- tensorboardX 2.6.2.2
- tqdm 4.66.1
- scikit-image 0.22.0
- numpy 1.22.4
- pandas 2.1.4
- matplotlib 3.8.2
- opencv-python 4.9.0.80
- SimpleITK 2.3.1

## Usage
### Dataset Acquisition
- LA Dataset: [LA Dataset Link](https://github.com/yulequan/UA-MT/tree/master/data)
- kits19 Dataset: [kits19 Dataset Link](https://paperswithcode.com/dataset/kits19)
- Lits Dataset: [Lits Dataset Link](https://paperswithcode.com/dataset/lits17)
### Model Files
The model files can be downloaded from the following link:
- [Model Files](https://pan.baidu.com/s/19qFfX5eveS_yqRTK8s5Wuw?pwd=02ex)
  - Extraction Code: `02ex`

### Train
python ./code/utils/registration   #Label registration
python ./code/train.py  # Training on  dataset
### Test
python ./code/test.py  # Testing on  dataset

## Acknowledgements

We would like to express our gratitude to the Desco and Bidirectional Copy-Paste (BCP) teams for their inspiring work and valuable contributions. Their provided code and assistance have significantly influenced our project and helped shape its development. Thank you for your support and the foundational work that made this research possible.

