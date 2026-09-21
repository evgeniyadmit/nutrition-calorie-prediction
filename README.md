
#  Multimodal Calorie Prediction
### Мультимодальная модель для предсказания калорийности блюд

[English](#english) · [Русский](#русский)

---

<a id="english"></a>

## 🇬🇧 English

### Overview

A deep learning model for predicting the total caloric content of a dish using three sources of information:

- 📷 dish image;
- 📝 ingredient list;
- ⚖️ portion mass.

The project combines Natural Language Processing and Computer Vision in a single multimodal regression architecture.

### 🎯 Result

**Test MAE: 38.77 kcal**

The target metric for the project was **MAE < 50 kcal**.

The final model successfully achieved the required quality threshold.

### 🧠 Architecture

The model consists of three branches:

- **DistilBERT** (`distilbert-base-uncased`) for ingredient text;
- **ResNet18**, pretrained on ImageNet, for dish images;
- **MLP** for portion mass.

Text representations are obtained using masked mean pooling over DistilBERT token embeddings.

The representations from all three branches are concatenated and passed to a regression head with a single output neuron.

```text
Ingredients ──────► DistilBERT ──► Text features ──┐
                                                   │
Dish image ───────► ResNet18 ─────► Image features ├──► Fusion ──► MLP ──► Calories
                                                   │
Portion mass ─────► MLP ──────────► Mass features ─┘
```

### ⚙️ Training

The model was trained using:

- PyTorch;
- `L1Loss`;
- AdamW optimizer;
- separate learning rates for pretrained encoders and the regression head;
- learning-rate scheduling;
- validation-based checkpointing;
- fixed random seed for reproducibility.

The official test split was kept separate from training and model selection.

### 🔎 Error Analysis

After inference, the five test samples with the largest absolute prediction errors were analyzed.

The largest errors occurred for several dishes containing calorie-dense ingredients such as pizza, almonds, and sausage.

One possible limitation is that the ingredient list indicates which ingredients are present but does not provide their exact quantities. An image also cannot always reveal factors such as oil, sauces, cheese, or exact ingredient proportions.

Detailed EDA, training history, visualizations, test evaluation and error analysis are available in [`notebook.ipynb`](notebook.ipynb).

---

<a id="русский"></a>

## 🇷🇺 Русский

### О проекте

Проект посвящён разработке нейронной сети для предсказания общей калорийности блюда на основе трёх источников информации:

- 📷 фотографии блюда;
- 📝 списка ингредиентов;
- ⚖️ массы порции.

Для решения задачи используется мультимодальная архитектура, объединяющая методы обработки естественного языка и компьютерного зрения.

Задача сформулирована как задача регрессии.

### 🎯 Результат

**MAE на тестовой выборке: 38.77 ккал**

Целевая метрика проекта составляла **MAE < 50 ккал**.

Таким образом, итоговая модель достигла требуемого качества.

### 🧠 Архитектура

Модель состоит из трёх основных ветвей:

- **DistilBERT** (`distilbert-base-uncased`) обрабатывает список ингредиентов;
- предобученная **ResNet18** извлекает признаки из фотографии блюда;
- небольшой **MLP** обрабатывает массу порции.

Для получения единого представления текста используется masked mean pooling по токенам DistilBERT.

После обработки признаки трёх модальностей объединяются и передаются в регрессионную часть модели. Финальный слой содержит один выходной нейрон без функции активации.

```text
Ингредиенты ──────► DistilBERT ──► Текстовые признаки ─┐
                                                       │
Изображение ──────► ResNet18 ────► Признаки изображения├──► Объединение ──► MLP ──► Калорийность
                                                       │
Масса порции ─────► MLP ─────────► Числовые признаки ──┘
```

### ⚙️ Обучение

При обучении использовались:

- PyTorch;
- функция потерь `L1Loss`;
- оптимизатор AdamW;
- разные значения learning rate для предобученных энкодеров и новых слоёв;
- планировщик learning rate;
- сохранение лучшей модели по validation MAE;
- фиксированный seed для воспроизводимости.

Официальная тестовая выборка не использовалась для обучения и выбора лучшей модели.

### 🔎 Анализ ошибок

После получения предсказаний были проанализированы пять объектов тестовой выборки с наибольшей абсолютной ошибкой.

Среди наиболее сложных примеров встречались блюда с калорийными компонентами, включая пиццу, миндаль и колбасные изделия.

Одна из возможных причин крупных ошибок заключается в том, что список ингредиентов содержит информацию об их наличии, но не сообщает точное количество каждого компонента. По фотографии также сложно определить количество масла, соусов, сыра и точные пропорции ингредиентов.

Подробный EDA, процесс обучения, графики, итоговая оценка и анализ ошибок находятся в [`notebook.ipynb`](notebook.ipynb).

---

## 📁 Repository Structure / Структура репозитория

```text
nutrition-calorie-prediction/
├── notebook.ipynb
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── README.md
└── scripts/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── model.py
    └── utils.py
```

The dataset is not included in the repository due to its size.

Датасет не включён в репозиторий из-за его размера.

---

## 🚀 Run / Запуск

Install dependencies / Установите зависимости:

```bash
pip install -r requirements.txt
```

Run training / Запустите обучение:

```bash
python train.py --config scripts/config.py
```

Dataset paths and training parameters can be configured in `scripts/config.py`.

Пути к данным и параметры обучения можно изменить в `scripts/config.py`.

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Transformers` · `DistilBERT` · `ResNet18` · `torchvision` · `pandas` · `NumPy` · `Matplotlib`

---

## 📚 Project Context / О проекте

🇬🇧 The project was developed as part of a Deep Learning course and focuses on building a reproducible multimodal training and evaluation pipeline.

🇷🇺 Проект разработан в рамках курса по Deep Learning и посвящён построению воспроизводимого пайплайна обучения и оценки мультимодальной нейронной сети.
