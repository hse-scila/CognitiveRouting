# Cognitive Routing
This is a repository of a "Cognitive Routing of Steering-Vector Experts for Mathematical Reasoning in Language Models" paper.
## Guide
In the folder [code/](code/) you can find the main [jupyter notebook](code/CognitiveRouting.ipynb) of the project. Run to reproduce the main results.
In the folder [data](data) there is a test [dataset](data/test_all.xslx) that was used to obtain final [results](data/results). Other datasets were used to train and infer steering vectors.
In the folder [steering/](steering/) you can find trained steering vectors for 3 types of differential equations for 4 models.
To reproduce steering vectors, use inference and tune notebooks in the [code](code/) folder and [datasets](data/datasets).  
## Installation 

```bash
git clone https://github.com/hse-scila/CognitiveRouting
```
## Models
Following models were used in the project. To reproduce experiments, download this models from Hugging face website and put them inside [models/](models/) folder:
- [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B)
- [Open-Reasoner-Zero-1.5B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B)
- [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
- [Gemma-3-1B](https://huggingface.co/google/gemma-3-1b-it)
