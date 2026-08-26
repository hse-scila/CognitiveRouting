# Differential Equation Type Classifier

ML-классификатор для определения типа дифференциального уравнения по текстовой записи.

Сейчас поддерживаемые классы:

- `unhomogenous`
- `polinomial`
- `separable`

## Формат датасета

По умолчанию скрипт ожидает CSV-файл с колонками:

```csv
equation,label
"dy/dx = x*y",separable
"y'' + y = sin(x)",unhomogenous
"dy/dx = x^2 + y^3",polinomial
```

Если колонки называются иначе, укажи их через `--text-column` и `--label-column`.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Обучение

```bash
train-equation-classifier --data path\to\dataset.csv
```

Полный пример:

```bash
train-equation-classifier ^
  --data data\equations.csv ^
  --text-column equation ^
  --label-column label ^
  --model-out models\equation_classifier.joblib
```

После обучения модель будет сохранена в `models/equation_classifier.joblib`.

## Предсказание одного уравнения

```bash
predict-equation-type ^
  --model models\equation_classifier.joblib ^
  --equation "dy/dx = x*y"
```

## Предсказание для CSV-файла

```bash
predict-equation-type ^
  --model models\equation_classifier.joblib ^
  --input data\new_equations.csv ^
  --text-column equation ^
  --output predictions.csv
```

## Почему такая модель

Для старта используется `TF-IDF` по символьным n-граммам и логистическая регрессия.
Для математических выражений это хороший baseline: модель видит локальные шаблоны вроде `dy/dx`, `y'`, `sin(x)`, `x^2`, `*y`, скобки, степени и знаки операций.

Если точности не хватит, следующим шагом можно добавить:

- нормализацию LaTeX/SymPy;
- ручные признаки по структуре уравнения;
- transformer-модель для формул;
- валидацию на отдельном holdout-наборе.
