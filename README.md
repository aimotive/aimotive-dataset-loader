# aiMotive Multimodal Dataset

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
