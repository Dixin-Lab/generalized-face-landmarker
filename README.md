# Generalizable Face Landmarking Guided by Conditional Face Warping (CVPR 2024)
This is the official repository for the following paper:

>**Generalizable Face Landmarking Guided by Conditional Face Warping**  [[paper]](https://openreview.net/pdf?id=wB2R7QQncw) [[project page]](https://plustwo0.github.io/project-face-landmarker/)<br>
 <br>Jiayi Liang, Haotian Liu, Hongteng Xu, Dixin Luo<br>
 Accepted by CVPR 2024.


![Scheme](/assets/scheme.png "Learning Scheme")

# Install

```commandline
pip install -r requirements.txt
```

# Model
Our proposed framework mainly contains two parts: **face warper** and **landmark detector**. 
They are trained in an alternative optimization framework. 
- The face warper aims to deform real human faces according to stylized facial images, generating warped faces and corresponding warping fields. 
- The face landmarker, as the key module of the warper, predicts the ending points of the warping fields and thus provides us with pseudo landmarks for the stylized facial images. 

In our implmentation, we employ SLPT as our backbone and locate the model in ```Landmark2``` folder. 
For the reproduction on other detectors, substitute the ```Landmark2``` folder with target model and make modifications in ```train.py```.

# Prepare

## Data



### Source Domain
Download images and annotations of 300-W from [ibug](https://ibug.doc.ic.ac.uk/resources/300-W/).

We select frontal faces from the trainset of 300W as our training data, and list of image path ```300W_frontal_train_list.txt``` can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LPYKxb2e-7a7Ovy2tPBPKad74HxxGZRG?usp=sharing).

### Target Domain
- Download CariFace according to [CariFace Dataset](https://github.com/Juyong/CaricatureFace).
The split of training and testing set can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1_oQhYCOJastWGhy6tFDUEivgj5GMLUCe?usp=sharing).

- Other domains like Artistic-Faces can be retrieved from [Link](https://github.com/papulke/face-of-art). 
The split of training and testing set can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1XN8_DfrbT7rfWbxx0oHpzqXOpGPCWcn7?usp=sharing).


# Train

## Load Pretrained Model
Download source-pretrained weights ```model_best.pth``` from [Google Drive](https://drive.google.com/file/d/1bkweD9atM-ON68IHRkiEr2AnkLHpKMGA/view?usp=sharing) and move it into folder ```Landmark2```.


## Training Begin!
 
```python
python train.py --src_data path/to/source/data --tgt_data path/to/target/data --pretrain_path path/to/pretrained/checkpoint
```


# Inference
Download our [model](https://drive.google.com/file/d/14AO7RmSYnSQ1fluq0cJo09Dia9Pakgp_/view?usp=sharing) and test on the CariFace by running:
```python
python test.py --checkpoint path/to/model/weights
```

Further, to test on ArtiFace, download [checkpoint](https://drive.google.com/file/d/1HNadvxvmYPX0KdvgWnBaldf5C1elHQ8i/view?usp=sharing) and inference:
```python
python test_Artistic.py --checkpoint path/to/model/weights
```

# Citation

# Aknowledgement
