# Development Environment

## Create the Conda Environment

```shell script
conda env create -f devenv/environment.yaml
```

or install the packages manually

```shell script
conda create -n tensorclan python=3.8
conda install -c pytorch pytorch torchvision cudatoolkit=10.2
conda install -c anaconda sphinx sphinx_rtd_theme pyyaml
conda install -c conda-forge albumentations tensorboard jupyter recommonmark
```

### Notes

exporting environment

```shell script
conda env export --no-builds > devenv/environment.yml
```