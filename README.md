# DFENet

（Brief description of the project）

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
- VT821, VT1000, VT5000: Baidu cloud disk [link](https://pan.baidu.com/s/1Vv6mYz4RL2VnwWwZWKLHyA), fetch code (djas); 
- VI-RGBT1500: [link](https://github.com/huanglm-me/VI-RGBT1500)
## Pertrain Models

|          |Params(M)| FLOPs(G)| Input size |  Backbone   | Link |
|----------|---------|---------|------------|-------------|------|
| DFENet-m | 149.2   |  139.7  |  384x384   | CDFFormer-m |      |
| DFENet-b | 264.6   |  238.0  |  384x384   | CDFFormer-b |      |
| DFENet-h | 149.8   |  248.3  |  512x512   | CDFFormer-m |      |

## Results

## Evaluation Metrics Toolbox
- The Evaluation Metrics Toolbox is available here: [link](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Acknowledgements
- Thanks to all the seniors who put in the effort.

## Contact Us
If you have any questions, please contact us ( lvpengfei1995@163.com ).
