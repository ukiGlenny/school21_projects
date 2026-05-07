# school21_projects

Три проекта, выполненных в ходе обучения в Школе 21 (Сбер). Основной фокус — практическое применение моделей ML, библиотек и инструментов анализа данных.

---

## ML1. Прогнозирование цен на аренду жилья (`renta.ipynb`)

**Инструменты и библиотеки:**  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (LinearRegression, DecisionTreeRegressor, PolynomialFeatures, train_test_split)

**Модели:**
- Линейная регрессия (с полиномиальными признаками до 10-й степени)
- Дерево решений (DecisionTreeRegressor)
- Наивные модели (предсказание среднего / медианы)

**Основные действия:**
- Обработка выбросов (IQR, процентили)
- Создание полиномиальных признаков
- Кодирование категориальных переменных
- Сравнение моделей по MAE, RMSE, R²

**Результат:** Дерево решений показало лучшее качество (MAE = 754).

---

## ML2. Линейные модели с регуляризацией (`supervised_learning.ipynb`)

**Инструменты и библиотеки:**  
`scikit-learn` (LinearRegression, Ridge, Lasso, ElasticNet, MinMaxScaler, StandardScaler, PolynomialFeatures), `numpy`, `pandas`, `matplotlib`

**Модели:**
- Линейная регрессия (в том числе собственная реализация через SGD)
- Ridge (L2-регуляризация)
- Lasso (L1-регуляризация)
- ElasticNet (L1 + L2)

**Основные действия:**
- Реализация `MyLinearRegression` с градиентным спуском и регуляризацией
- Нормализация данных (MinMaxScaler, StandardScaler)
- Логарифмическое преобразование целевой переменной
- Удаление выбросов
- Отбор признаков через веса Lasso, корреляцию Пирсона, SHAP

**Результат:** Lasso эффективен для отбора признаков, Ridge — для стабилизации решения.

---

## ML3. Валидация и отбор признаков (`validation.ipynb`)

**Инструменты и библиотеки:**  
`scikit-learn` (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, permutation_importance), `optuna`, `shap`, `numpy`, `pandas`

**Методы валидации (реализованы вручную и сравнены с sklearn):**
- K-Fold
- GroupKFold
- StratifiedKFold
- TimeSeriesSplit
- Leave-One-Out (теория)

**Методы отбора признаков:**
- Корреляция Пирсона
- Permutation Importance
- SHAP (Shapley values)
- Lasso-коэффициенты

**Оптимизация гиперпараметров:**
- Grid Search
- Random Search
- Optuna (Bayesian optimization)

**Модель для оптимизации:** ElasticNet

**Результат:** Optuna показала лучший баланс скорости и качества; StratifiedKFold оптимален для несбалансированных данных.

---

## Общий стек технологий

| Категория | Инструменты |
|-----------|--------------|
| Язык | Python 3.13 |
| Анализ данных | pandas, numpy |
| Визуализация | matplotlib, seaborn |
| Модели | LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeRegressor |
| Предобработка | MinMaxScaler, StandardScaler, PolynomialFeatures, train_test_split |
| Валидация | KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit |
| Оптимизация | GridSearch, RandomSearch, Optuna |
| Интерпретация | SHAP, permutation_importance |

---

*Все проекты выполнены мной самостоятельно в рамках обучения в Школе 21 (Сбер).*
