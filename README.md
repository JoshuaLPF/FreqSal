# DFENet
## Deep Fourier-embedded Network for Bi-modal Salient Object Detection [[paper (an initial version)]](https://arxiv.org/abs/2411.18409)
- **April 29, 2024**  
  The paper is undergoing peer review. The code will be released upon acceptance of the paper.
- ![Framework](https://github.com/JoshuaLPF/DFENet/blob/main/Figure/framework.png)
- In this project, we proposed the deep Fourier-embedded network (DFENet), a purely Fourier-based model aimed at solving the high-resolution bi-modal inputs and feature fusion while minimizing memory consumption of GPU, outperforming existing state-of-the-art bi-modal salient object detection (BSOD) models on four RGB-T BSOD benchmark datasets. **To the best of our knowledge, this is the first Fourier-based supervised model in a series of salient object detection tasks.**
- Please cite our paper if you find it useful for your research.
```
@article{lyu2024deep,
  title={Deep Fourier-embedded Network for Bi-modal Salient Object Detection},
  author={Lyu, Pengfei and Yu, Xiaosheng and Wu, Chengdong and Rajapakse, Jagath C},
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
- We conducted experiments to evaluate our DFENet on the [VT821, VT1000, VT5000](https://github.com/lz118/RGBT-Salient-Object-Detection), and [VI-RGBT1500](https://github.com/huanglm-me/VI-RGBT1500) datasets. Please click for the corresponding dataset.
 
## Pre-trained Weights of DFENet

  Resolution  | Backbone | weights
 ---- | ----- | ------  
 384x384 | [CDFFormer-m36](https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_m36.pth) | [Link](https://pan.baidu.com/s/1j_u9YGr-9zwNOJHJ9WCuqQ) (fetch code: kbpy)  
 512x512 | [CDFFormer-m36](https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_m36.pth) | [Link](https://pan.baidu.com/s/1j_u9YGr-9zwNOJHJ9WCuqQ) (fetch code: kbpy)  

## Results
The results of our DFENet can be found at [link](https://pan.baidu.com/s/19aWbiGBD6AqWrP0e_PwYWw), fetch code(1ryz).

## Evaluation Metrics Toolbox
- The Evaluation Metrics Toolbox is available here: [link](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Acknowledgements
- Thanks to all the seniors and projects (*e.g.*, [DFFormer](https://github.com/okojoalg/dfformer), [FFL](https://github.com/EndlessSora/focal-frequency-loss), [UHDFour](https://li-chongyi.github.io/UHDFour/), [MGAI](https://github.com/huanglm-me/VI-RGBT1500), and [SwinNet](https://github.com/okojoalg/dfformer)).

## Contact Us
If you have any questions, please contact us (lvpengfei1995@163.com).
