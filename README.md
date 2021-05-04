# Pytorch insightface
MTCNN detector + iresnet embedder
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

<details>
<summary>Installation</summary>

### Create virtual env use conda (recommend)

```
conda create -n face_detection python=3
source activate face_detection
```

### Installation dependency package

```bash
pip install -r requirements.txt
```

If you have gpu on your machine, you can follow the [official pytorch installation](https://pytorch.org/) and install pytorch gpu version.

### Compile the cython code
Compile with gpu support
```bash
python setup.py build_ext --inplace
```
Compile with cpu only
```bash
python setup.py build_ext --inplace --disable_gpu 
```

### Also, you can install mtcnn as a package
```
python setup.py install
```
</details>

## Basic Usage

```python
from embeddings import get_models, get_embeddings 

detector, embedder = get_models()

img_path = 'tests/asset/images/roate.jpg'
features = get_embeddings(img_path, detector, embedder)
print(features.shape)
print(features[0,:5])
```

## Doc
[Train your own model from scratch](./doc/TRAIN.md)

## Tutorial

[Detect step by step](./tutorial/detect_step_by_step.ipynb).

[face_alignment step by step](./tutorial/face_align.ipynb)

[face_embeddings](./tutorial/get_embeddings.ipynb)

