# Jointly Learn the Base Clustering and Ensemble for Deep Image Clustering(JDCE)
This is the code for the paper "Jointly Learn the Base Clustering and Ensemble for Deep Image Clustering" (ICME 2024)
![structure.png](/structure.png)

# Dependency
* python==3.10.9
* numpy==1.24.2
* Pillow==9.2.0
* scikit_learn==1.1.1
* scipy==1.10.0
* torch==1.12.0
* torchvision==0.13.0
* tqdm==4.64.0

# Datasets
CIFAR-10, CIFAR-100, STL-10 will be automatically downloaded by Pytorch.

For ImageNet-10 and ImageNet-dogs, the description of subsets can find [here](https://github.com/Yunfan-Li/Contrastive-Clustering).
The folder of these two ImageNet datasets should be like this:
```
Datasets
    ├──  IMAGENET-10
    │   ├──data.npy
    │   ├──label.npy
    ├──  IMAGENET-DOG
    │   ├──data.npy
    │   ├──label.npy

```

# Usage

## Training
To train our model, run the following script. 
```bash
$ python main.py
```

## Test
Once the training is completed, there will be a saved model in folder ```saved_model/dataset_name```. To test the trained model, run
```bash
$ python test.py
```
We provide some trained models, you can download here or find them in folder mentioned above.

## Citation
```
@inproceedings{Chen_2024_Icme,
    author={Chen Liang, Zhiqian Dong, Sheng Yang, Peng Zhou},
    title={Jointly Learn the Base Clustering and Ensemble for Deep Image Clustering},
    booktitle={International Conference on Multimedia and Expo},
    year={2024},
}
```
---

## Credit
Some parts of this code (e.g., network) are based on [CC](https://github.com/Yunfan-Li/Contrastive-Clustering) repository.
