# FreqSal
## Deep Fourier-embedded Network for RGB and Thermal Salient Object Detection [[IEEE]](https://ieeexplore.ieee.org/document/11230613)[[arXiv]](https://arxiv.org/abs/2411.18409) 
- **Nov. 4, 2025**  
  The paper has been accepted by TCSVT2025
- **Apr. 29, 2024**  
  The paper is undergoing peer review. The code will be released upon acceptance of the paper.
- ![Framework](https://github.com/JoshuaLPF/FreqSal/blob/main/Figure/framework.png)
- In this project, we proposed the deep Fourier-embedded network, namely FreqSal, a purely Fourier-based model aimed at solving the high-resolution bimodal inputs and feature fusion while minimizing memory consumption of GPU, outperforming existing state-of-the-art bimodal salient object detection (SOD) models on four RGB-T, five RGB-D, and one RGB-D-T SOD benchmark datasets. **To the best of our knowledge, this is the first Fourier-based supervised model in a series of SOD tasks.**
- Please cite our paper if you find it useful for your research.
```
@article{lyu2024deep,
  title={Deep Fourier-embedded Network for RGB and Thermal Salient Object Detection},
  author={Lyu, Pengfei and Yeung, Pak-Hei and Yu, Xiaosheng and Wu, Chengdong and Rajapakse, Jagath C},
  journal={arXiv preprint arXiv:2411.18409},
  year={2024}
}
```
## Requirements

List of prerequisites or required libraries for the project to run:

- Pytorch 2.0.0
- Cuda 11.8
- Python 3.8 or higher
- tensorboardX
- opencv-python
- timm == 0.5.4
- thop
- numpy

## Datasets
- We conducted experiments to evaluate our model on the [VT821, VT1000, VT5000](https://github.com/lz118/RGBT-Salient-Object-Detection), and [VI-RGBT1500](https://github.com/huanglm-me/VI-RGBT1500) datasets for the RGB-T SOD task, and on the [NLPR, NJUD, DUT-RGBD, SIP, and STERE](https://github.com/jiwei0921/RGBD-SOD-datasets) datasets for the RGB-D SOD task. Please click for the corresponding dataset.
 
## Pre-trained Weights of FreqSal

  Resolution  | Backbone | Tpye | weights
 ---- | ----- | ------ | ------ 
 384 x 384 | [CDFFormer-m36](https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_m36.pth) | RGB-T | [Link](https://pan.baidu.com/s/1NMvuPohsT1URkI529G013Q?pwd=umm4) 
 512 x 512 | [CDFFormer-m36](https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_m36.pth) | RGB-T | [Link]()
 384 x 384 | [CDFFormer-m36](https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_m36.pth) | RGB-D | [Link]()
 

## Results
- The RGB-T and RGB-D results of our model can be found at [link](https://pan.baidu.com/s/1tJMnR8cF_xH3i2aDy_662g?pwd=vuj4).
- ![RGB-T results](https://github.com/JoshuaLPF/FreqSal/blob/main/Figure/RGBT.png)
- ![RGB-D results](https://github.com/JoshuaLPF/FreqSal/blob/main/Figure/RGBD.png)

## Evaluation Metrics Toolbox
- The Evaluation Metrics Toolbox is available here: [link](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Acknowledgements
- Thanks to all the seniors and projects (*e.g.*, [DFFormer](https://github.com/okojoalg/dfformer), [FFL](https://github.com/EndlessSora/focal-frequency-loss), [UHDFour](https://li-chongyi.github.io/UHDFour/), [MGAI](https://github.com/huanglm-me/VI-RGBT1500), and [SwinNet](https://github.com/okojoalg/dfformer)).

## Contact Us
If you have any questions, please contact us (lvpengfei1995@163.com).
