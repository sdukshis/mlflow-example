# Пример работы MLFLow c FeatureStorage

## Подготовка окружения

Создайте виртуальное окружение на версии python3.10
```
python3.10 -m venv .venv
source .venv/bin/activate.sh
```

Установите пакет с примером и зависимостями
```
pip install -e .
```

Запустите MLFlow и необходимые ему компоненты с помощью docker compose
```
docker compose up -d
```

## Обучение и сохранение модели

Для запуска обучения и сохранения модели в MLFlow используйте скрипт
```
./example/train.py
```

Для инференса модели можно исопльзовать test.ipynb подставив нужный run_id.
Или запустив через mlflow serve
```
mlflow models serve -m "runs:/<run_id>/models/model" --port 8081
```

Проверить работу модели можно с помощью команды
```
http http://localhost:8081/invocations dataframe_records:='[{"f0": 0, "f1": 1, "f2": 2}]'
```

При этом в логах mlflow serve будет логироваться обогащенные параметры.

Значения параметров берутся из таблицы features в БД mlflow_database в MySQL.

Можно поменять значения в таблице features и увидеть эффект при инференсе в логах.
```
echo "UPDATE features SET value = 303 WHERE name = 'market'" | mysql -h 127.0.0.1 -u mlflow_user -p  -b  mlflow_database
```