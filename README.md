# aiMotive Multimodal Dataset Loader

## Download
The dataset can be downloaded from this [repository](https://github.com/aimotive/aimotive_dataset).

## Installation
The repository has been tested on Ubuntu with Python 3.8. Currently no Windows support is available.
### Create a conda environment
```
conda create --name aimdataset python=3.8
conda activate aimdataset
```

### Clone repository
```
git clone https://github.com/aimotive/aimotive-dataset-loader.git
cd aimotive-dataset-loader
```

### Install requirements
```
pip install -r requirements.txt
```

## Examples
The repository includes a small sample dataset with 50 keyframes. The examples demonstrate how the data can be rendered
and loaded to PyTorch framework.

### Run rendering example
```
PYTHONPATH=$PYTHONPATH: python examples/example_render.py
```

### Run PyTorch loader example
#### Install torch
```
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Run script
```
PYTHONPATH=$PYTHONPATH: python examples/pytorch_loader.py
```

## Cite our work
If you use this code or aiMotive Multimodal Dataset in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{matuszka2022aimotivedataset,
  title={aiMotive Dataset: A Multimodal Dataset for Robust Autonomous Driving with Long-Range Perception},
  author={Matuszka, Tamás and Németh, Gabor and Varga, Bálint and Kunsági-Máté, Sándor and Barton, Iván and Hajas, Péter and Szeghy, Dávid and Pető, Levente},
  journal={arXiv preprint arXiv:XXXXXXXX},
  year={2022}
}
```
