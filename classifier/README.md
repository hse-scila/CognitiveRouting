# Differential Equation Type Classifier

The classifier used by `code/CognitiveRouting.ipynb` is the PyTorch
`MathTextCNN` checkpoint in `pytorch_equation_classifier.pt`. Its labels are
`polynomial`, `separable`, and `unhomogenous`.

The CNN uses 96-dimensional token embeddings, three convolution kernels
(3, 5, 7), 128 filters per kernel, max pooling, and dropout 0.25. The
preprocessing and checkpoint structure match the routing notebook.

## Clean datasets

- `data/combined_equations.xlsx` contains 657 unique training-source equations.
- `data/combined_equations.csv` and `data/combined_equations_with_split.csv`
  contain the same records and a fixed, stratified split: 591 train / 66 valid.
- `data/dataset_summary.xlsx` summarizes these cleaned records.
- `data/test_equations.csv` is a snapshot of the 244 equations in
  `../data/datasets/test_all.xlsx`, with class labels and original Excel row numbers.

The external test is not included in the combined training workbook. Validation
is drawn only from training-source data and is used to select the epoch. Vocabulary,
sequence length, and class weights are calculated from the train split only.

Training data takes priority when the source datasets overlap. The 19 previously
removed answer-conflict records and 3 overlapping train records have been restored
to the separable source workbook. Those 3 overlap records represent 2 distinct
equations, which have been removed from the current test datasets and experiment
results. The source train workbook has 271 records; the separable test has 64.

The classifier uses one record per normalized input equation. Restoring the source
records adds 10 unique inputs. Existing split assignments are preserved; 9 added
inputs enter train and 1 enters validation, selected with seed 42. Current and
historical test equations remain excluded, except the 2 equations explicitly
reassigned to training. Source filenames refer to the actual repository datasets.
Details and removed test/result records are in [reports](reports/README.md).

## Install and retrain

Run from this directory:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
retrain-pytorch-classifier
```

Alternatively, open `differential_equation_pytorch_classifier.ipynb`.
The command trains from scratch with seed 42 and the original CNN hyperparameters,
selects the checkpoint by validation macro F1, and evaluates the saved model on
the external test once. It updates `pytorch_equation_classifier.pt` and writes:

- `reports/evaluation.json`: metrics, confusion matrix, configuration, and input hashes;
- `reports/test_predictions.csv`: predictions and class probabilities for every test row;
- `reports/training_history.csv`: training and validation history.

Training refuses duplicate inputs, test-source rows, overlap with the current or
reserved historical test equations, and stale external-test snapshots. If the external XLSX
changes, refresh and audit the CSV snapshot before retraining.

The revised run on 2026-10-08 scored 244/244 correct (accuracy 1.0, macro F1 1.0)
on the current external test. This measures classification of this dataset;
it does not evaluate equation solutions or steering-vector performance.

## Baseline implementation

The existing `train-equation-classifier` and `predict-equation-type` commands
implement a separate scikit-learn TF-IDF / logistic-regression baseline. They are
not the classifier loaded by the routing notebook.

The PyTorch retraining command above is the reproducible path for the paper's model.
