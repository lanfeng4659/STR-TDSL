# Scene Text Retrieval via Joint Text Detection and Similarity Learning （CVPR2021）

This is the code of "Scene Text Retrieval via Joint Text Detection and Similarity Learning". For more details, please refer to our [CVPR2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Scene_Text_Retrieval_via_Joint_Text_Detection_and_Similarity_Learning_CVPR_2021_paper.pdf).

This repo is inherited from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and follows the same license.

## Chinese Street View Text Retrieval Dataset (CSVTR)

CSVTR consists of 23 pre-defined query words in Chinese and 1667 Chinese scene text images collected from the Google image search engine. Each image is annotated with its corresponding query word among the 23 pre-defined Chinese query words. 

CSVTR could be downloaded from [baidu disk](https://pan.baidu.com/s/1CqKZ7zZL5U9uSsyBw0l3ag)(asjw) or [google driver](https://drive.google.com/file/d/1aC7_a3_2k7skeTT3EeM54UO76jpx9Pm4/view?usp=sharing).

## Trained models
The trained models could be downloaded from [baidu disk](https://pan.baidu.com/s/1vLR4EzXYyof-l69b621jog)(legq).

## Evaluation
### 1. prepare datasets
An example of the path of test images: ./datasets/IIIT_STR_V1.0/imgDatabase/img_000846.jpg
### 2. evaluate
### run ```sh test.sh```

## Citing the related works

Please cite the related works in your publications if it helps your research:

    @InProceedings{Wang_2021_CVPR,  
      author    = {Wang, Hao and Bai, Xiang and Yang, Mingkun and Zhu, Shenggao and Wang, Jing and Liu, Wenyu},  
      title     = {Scene Text Retrieval via Joint Text Detection and Similarity Learning},  
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
      month     = {June},  
      year      = {2021},  
      pages     = {4558-4567}  
    }

