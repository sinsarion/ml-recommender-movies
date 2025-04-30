# Установка и запуск проекта

## Шаг 1. Клонирование репозитория
```bash
git clone https://github.com/sinsarion/ml-recommender-movies.git
cd ml-recommender-movies
```

## Шаг 2. Создание виртуального окружения Python
```bash
python -m venv venv
source venv/bin/activate  # для Linux/macOS
venv\Scripts\activate    # для Windows
pip install --upgrade pip
pip install -r src/requirements.txt
```

## Шаг 3. Запуск Docker-compose
```bash
docker-compose up -d
```
Это запустит:
- PostgreSQL
- MongoDB
- Redis
- FastAPI - ML API сервиса рекомендаций

## Шаг 4. Скачивание и распаковка датасета MovieLens 32M
```bash
cd datasets
python etl_movielens_32m.py
```
Этот скрипт:
- скачивает архив MovieLens 32M
- распаковывает в папку `movielens_32m/ml-32m`
- импортирует датасет MovieLens 32M в PostgreSQL
- удаляет .zip и временные файлы

## Шаг 5. Импорт оценок в MongoDB
```bash
cd datasets
python etl_movielens_32m_ratings_to_mongo.py
```

## Шаг 6. Генерация матриц для LightFM
```bash
cd src/ml_lightfm
python generate_interaction_matrix.py
python generate_item_features_all.py
```

## Шаг 7. Обучение LightFM
```bash
python train_lightfm_model.py
```
Этот скрипт:
- создает файл модели `artifacts/lightfm_model.pkl`
- обучает модель 30 эпох (если необходимо изменить количество эпох - можно поменять скрипт)

## Шаг 8. Генерация датасета для NeuMF (PyTorch)
```bash
python prepare_neumf_dataset.py
```
Этот скрипт:
- создает файл датасета neumf_dataset.pkl, необходимый для обучения модели NeuMF

## Шаг 9. Обучение NeuMF (PyTorch)
```bash
python train_neumf_model.py --epochs 30
```
Этот скрипт:
- создает файл модели `artifacts/neumf_model_v2.pth`
- обучает модель 30 эпох (опционально)

## Запуск API
После этого API готово к обработке запросов на `GET /recommendations` и `GET /recommendations/{user_id}`

---

[← Назад к README](README.md)

