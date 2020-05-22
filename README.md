# Image-Scaling Attacks & Defenses

This repository belongs to our publication:

---

Erwin Quiring, David Klein, Daniel Arp, Martin Johns and Konrad Rieck.
Adversarial Preprocessing: Understanding and Preventing Image-Scaling Attacks in Machine Learning.
*Proc. of USENIX Security Symposium*, 2020.

---

## Background
For an introduction together with current works on this topic, please visit
our [website](http://scaling-attacks.net).

<p align="center">
<img src="./example.jpg" width="458" height="190" alt="Principle of image-scaling attacks" />
</p>

In short, image-scaling attacks enable an adversary to manipulate images, such
that they change their appearance/content after downscaling. In
particular, the attack generates an image A by slightly
perturbing the source image S, such that its scaled version D
matches a target image T. This process is illustrated in the figure above.

## Getting Started
This repository contains the main code for the attacks and defenses. It has a
simple API and can be easily used for own projects. The whole project consists
of python code (and some cython additions).

### Installation
In short, you just need the following steps (assuming you have Anaconda).

Get the repository:
```
git clone https://github.com/EQuiw/2019-scalingattack
cd 2019-scalingattack/scaleatt
```
Create a python environment (to keep your system clean):
```
conda create --name scaling-attack python=3.6
conda activate scaling-attack
```
Install python packages and compile cython extensions:
```
pip install -r requirements.txt
python setup.py build_ext --inplace
```

Check the [README](./scaleatt/README.md) in the scaleatt directory for a
detailed introduction how to set up the project (in case of problems).

That's it. For instance, to run the tutorial, you can use (assuming you're
still in directory *scaleatt* and use BASH for ```$(pwd)```):
```
PYTHONPATH=$(pwd) python tutorial/defense1/step1_non_adaptive_attack.py
```

## Tutorial

### Jupyter Notebook
For a quick introduction, I recommend you to look
at [this jupyter notebook](./scaleatt/tutorial/jupyter_intro.ipynb).


### Main Tutorial
Check the directory *scaleatt/tutorial/* for a detailed tutorial how to run the
attacks and defenses.

The directory has the same structure as our evaluation. Each subdirectory
corresponds to the subsection from our paper:
- The directory *defense1* corresponds to experiments from Section 5.2 and 5.3
- The directory *defense2* corresponds to experiments from Section 5.4 and 5.5
  - Each subdirectory contains some python scripts that describe the API
  and the respective steps.

My recommendation: Open each file (in the order of the steps), and then use
a python console to run the code _step by step interactively_.
