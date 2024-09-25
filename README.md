<div align="center">
<h1>World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering</h1>
<a href='https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE'><img src='https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg'></a>
<a href='https://openreview.net/forum?id=9IVeNxXFV4'><img src='https://img.shields.io/badge/Paper-Openreview-red'></a>
  
Jiacong Wang<sup>1,2*</sup>, 
[Bohong Wu](https://scholar.google.com/citations?user=N6vypvkAAAAJ&hl=en)<sup>2*</sup>,
[Haiyong Jiang](https://people.ucas.ac.cn/~0054404)<sup>1</sup>, 
Xun Zhou<sup>2</sup>, 
[Xin Xiao](https://scholar.google.com/citations?user=CL-ZEdwAAAAJ&hl=zh-CN)<sup>2</sup>, 
[Haoyuan Guo](https://scholar.google.com/citations?hl=en&user=hql67boAAAAJ&view_op=list_works&sortby=pubdate) <sup>2</sup>
[Jun Xiao](https://scholar.google.com/citations?hl=zh-CN&user=9YbplfMAAAAJ)<sup>1</sup>, 

<sup>1</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences, <sup>2</sup>ByteDance Inc

</div>

## Todo(Comming Soon)
Data Generate Pipline(WIP)

Training Code(WIP)


## News and Updates


<!-- ## Selected Examples
<p align="center"><img width="100%" src="./images/cases.jpg"></p> -->

## Results
We provide results comparision for LLaVA-NEXT here. 
<p align="center"><img width="80%" src="./vqa.jpeg"></p>
<p align="center"><img width="80%" src="./grouding.jpeg"></p>


### 1. Customize base settings
Before training, **you need to customize some settings in the following table**. Otherwise, the code will use the default paths specified in [run.sh](LLaVA/scripts/general/run.sh). **When using multiple data sources, simply concatenate their paths with a space**.


| Setting            | Usage                                         |
|--------------------|-----------------------------------------------|
| `base_dir`         | Path saving root directory                    |
| `exp_name`         | Experiment name, associated with the saving path |
| `pretrain_json`    | Pretrain JSON data                            |
| `pretrain_imagedir`| Pretrain data image directory                 |
| `finetune_json`    | Finetune JSON data                            |
| `finetune_imagedir`| Finetune data image directory                 |


## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): the codebase we evaluate our model.

Thanks a lot for their great works.

## Citation
```bibtex
@inproceedings{
anonymous2024world,
title={World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering},
author={Anonymous},
booktitle={Submitted to ACL Rolling Review - June 2024},
year={2024},
url={https://openreview.net/forum?id=9IVeNxXFV4},
note={under review}
}
```












