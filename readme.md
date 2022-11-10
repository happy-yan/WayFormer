# Wayformer

## Installation

```shell
conda create -n wayformer python=3.8
conda activate wayformer
git clone git@github.com:argoai/argoverse-api.git
```

- change directory to `argoverse-api`
- Download `hd_maps.tar.gz` from [this website](https://www.argoverse.org/av1.html#download-link) and extract into the root directory of the repo.

Your directory structure should look something like this:

```
└── argoverse
    └── data_loading
    └── evaluation
    └── map_representation
    └── utils
    └── visualization
└── map_files
└── license
```

- Download Argoverse-Forecasting dataset, extract into any directory.

- Install argoverse package (in `argoverse-api` root directory)

```shell
sudo snap install cmake  #ignore if cmake is installed
pip install -e .
```

Make sure that you can run `import argoverse` in python.

Install packages according to requirements.txt

## Run

change directory to `wayformer`
```shell
mkdir configs
python src/init_config.py
```

configurate your experiment by modifying configs/*.json file.

run experiment

```shell
python src/run.py
```