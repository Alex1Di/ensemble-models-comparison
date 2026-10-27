# Ensemble Models Comparison

Учебный ML-проект курса KarpovCourse HardDA: сравнение ансамблевых моделей для предсказания логарифма цены автомобиля по данным объявлений.

## Что внутри

`Ansambles RF + GB.ipynb` — полный пайплайн:

- **Выгрузка данных** из ClickHouse (`live_adverts`, `user_passports`).
- **Feature engineering**: регион, платформа, бренд, модель, год, тип продавца.
- **Предподготовка**: `OneHotEncoder` + `TargetEncoder` через `ColumnTransformer`.
- **Две версии данных**:
  - **slim** — без `auto_model`;
  - **full** — с `auto_model`.
- **Модели**:
  - Decision Tree (базовая);
  - Random Forest;
  - XGBoost;
  - CatBoost.
- **Кросс-валидация** (`cross_validate`) и **подбор гиперпараметров** (`GridSearchCV`).
- **Метрики**: R², MAE.
- **Анализ важности признаков**.

## Результаты

| Модель | R² (CV) | MAE (test) |
|---|---|---|
| Random Forest (full) | ~0.76 | — |
| XGBoost (full) | ~0.76 | — |
| CatBoost (full) | ~0.77 | 0.34 |

## Стек

- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost, catboost
- clickhouse-driver
- matplotlib, seaborn

## Как запустить

1. Установить зависимости:
   ```bash
   pip install pandas numpy scikit-learn xgboost catboost clickhouse-driver matplotlib seaborn