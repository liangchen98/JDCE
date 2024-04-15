# Jointly Learn the Base Clustering and Ensemble for Deep Image Clustering(JDCE)
This is the code for the paper "Jointly Learn the Base Clustering and Ensemble for Deep Image Clustering" (ICME 2024)
![structure.png](/structure.png)

# Dependency
* python==3.10.9
* numpy==1.24.2
* Pillow==10.3.0
* scikit_learn==1.1.1
* scipy==1.10.0
* torch==1.12.0
* torchvision==0.13.0
* tqdm==4.64.0

# Datasets
CIFAR-10, CIFAR-100 and STL-10 can be automatically downloaded by Pytorch.

For ImageNet-10 and ImageNet-dogs, the description of selected subsets can find [here](https://github.com/Yunfan-Li/Contrastive-Clustering).
The folder of these two ImageNet datasets should be like this:
```
data
  ├── ImageNet-10
  │     ├──data.npy
  │     ├──label.npy
  ├── ImageNet-DOG
  │     ├──data.npy
  │     ├──label.npy

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
We provide some trained models, please download [here](https://drive.google.com/drive/folders/1ewY3Ark5OuFRas3Nu7xi_VrNXXJS7g3L?usp=sharing).

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
