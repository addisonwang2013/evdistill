## EvDistill: Asynchronous Events to End-task Learning via Bidirectional Reconstruction-guided Cross-modal Knowledge Distillation (CVPR'21)

## Citation
if you find this resource helpful, please cite the paper as follows:

```bash
@article{wangevdistill,
  title={EvDistill: Asynchronous Events to End-task Learning via Bidirectional Reconstruction-guided Cross-modal Knowledge Distillation},
  author={Wang, Lin and Chae, Yujeong and Yoon, Sung-Hoon and Kim, Tae-Kyun and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Maintainer
[Lin Wang](https://sites.google.com/site/addisionlinwang/products-services?authuser=0)


## Setup

Download 

``` python
git clone  https://github.com/addisonwang2013/evdistill/
```

Make your own environment

```python
conda create -n myenv python=3.7
conda activate myenv
```

Install the requirements

```bash
cd evdistill

pip install -r requirements.txt
```

Download DDD17 test dataset from this link: [DDD17 test](https://sites.google.com/site/addisionlinwang/products-services?authuser=0)

Put the dataset to `./dataset/ddd17/`

Download the pretrained models from this link: [checkpoints](https://sites.google.com/site/addisionlinwang/products-services?authuser=0)

Put the checkpoints of event and aps segmentation networks into `./res/`

Visualize semantic segmentation results for both event and aps data:

```python
python submission.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

