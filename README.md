## EvDistill: Asynchronous Events to End-task Learning via Bidirectional Reconstruction-guided Cross-modal Knowledge Distillation (CVPR'21)

## Citation
If you find this resource helpful, please cite the paper as follows:

```bash
@article{wangevdistill,
  title={EvDistill: Asynchronous Events to End-task Learning via Bidirectional Reconstruction-guided Cross-modal Knowledge Distillation},
  author={Wang, Lin and Chae, Yujeong and Yoon, Sung-Hoon and Kim, Tae-Kyun and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
* Segmentic segmentation on DDD17 dataset

![image](https://user-images.githubusercontent.com/79432299/118368456-1df61b00-b5dd-11eb-87a7-54a1714628f9.png)

* Segmentic segmentation on MVSEC dataset

![image](https://user-images.githubusercontent.com/79432299/118368521-5a297b80-b5dd-11eb-8a98-b38c9879f014.png)


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

Download DDD17 test and MVSEC test dataset from this link: [DDD17 test and MVSEC test](https://drive.google.com/drive/u/1/folders/1zhQCr0r_xfButDnLLy1BoD2j7EGP18NQ)

* For DDD17 dataset, please put the dataset to `./dataset/ddd17/`
* For MVSEC dataset, please put the dataset to `./dataset/mvsec/`

Download the pretrained models from this link: [checkpoints](https://drive.google.com/drive/u/1/folders/10h7jhvzMX2dNKlbLXhp3ARRkhbUl8L7J)

* Put the checkpoints of event and aps segmentation networks for e.g. DDD17 dataset into `./res/`

Modify the ``` python configurations.py ``` in the `configs` folder with the relevant paths to the test data and checkpoints

* For the test data, *e.g.* DDD17, please assign the path to `./test_dir/ddd17/`
* For the checkpoint of aps nework, please assign the path to `./res/ddd17/ddd17_aps_ckpt.pth`
* For the checkpoint of event network, please assign the path to `./res/ddd17/ddd17_event_ckpt.pth`



Test the MIoU of event and aps segmentation networks:

```python
python test_evdistill.py
```

Visualize semantic segmentation results for both event and aps data:

```python
python visualize.py
```

## Note 

In this work, for convenience, the event data are embedded and stored as multi-channel event images, which are the paired with the aps frames. It is also possible to directly feed event raw data after embedding to the student network directly with aps frames.

## Acknowledgement
The skeleton code is inspired by [Deeplab-v3-Plus](https://github.com/CoinCheung/DeepLab-v3-plus-cityscapes/)

## License
[MIT](https://choosealicense.com/licenses/mit/)


