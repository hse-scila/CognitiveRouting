# Cognitive Routing
This is a repository of a "Cognitive Routing of Steering-Vector Experts for Mathematical Reasoning in Language Models" paper.
## Datasets
Every dataset used or obtained in the work can be found in the [data](data/) folder. Every dataset is a set of differential equations with solution and additional marks.

Datasets, that were used to train steering vectors are in [datasets](data/datasets/) subfolder and are diveded into 3 different equation types (inhomogeneous, polynomial and separable) as steering vectors were trained for different type of differential equation separately. In each of these you can find 3 datasets - train, train_short and test.

In the same [datasets](data/datasets/) folder you can find [test dataset](data/test_all.xlsx) used to obtain main results of the work. 

Main results of the work can be found in [results](data/results). They are divided by number of permitted output tokens - 500 or 2000.

## Code
[Here](code/) you can find all code as jupyter notebooks that were used in the work. 
- Main [nootebok](code/CognitiveRouting.ipynb) - it classifies the equation into 1 of 3 classes, then tries to give a solution by using AI model with our trained steerin vectors.
- Notebooks used to [tune](code/inhomogeneous_steering_tune.ipynb) and [infer](code/inhomogeneous_steering_tune.ipynb) steering vectors for inhomogeneous equations.
- Notebooks used to [tune](code/polynomial_steering_tune.ipynb) and [infer](code/polynomial_steering_tune.ipynb) steering vectors for polynomial equations.
- Notebooks used to [tune](code/separable_steering_tune.ipynb) and [infer](code/separable_steering_tune.ipynb) steering vectors for separable equations.

In the folder [steering](steering/) you can find trained steering vectors for 3 types of differential equations for 4 models.
## Installation 

```bash
git clone https://github.com/hse-scila/CognitiveRouting
```
No additional dependencies needed. Everything will be automatically installed in jupyter notebook or jupyter notebook will guide you to install any needed libraries via pip.  
## Guide
In the folder [code](code/) you can find the main [jupyter notebook](code/CognitiveRouting.ipynb) of the project. Run to reproduce the main results. Run other notebooks to reproduce training and inferring of steering vectors.

To reproduce steering vectors, use inference and tune notebooks in the [code](code/) folder and [datasets](data/datasets).  

## Models
Following models were used in the project. To reproduce experiments, download this models from Hugging face website and put them inside [models/](models/) folder:
- [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B)
- [Open-Reasoner-Zero-1.5B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B)
- [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
- [Gemma-3-1B](https://huggingface.co/google/gemma-3-1b-it)
