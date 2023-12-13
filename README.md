# Поиск похожих товаров с помощью FAISS

## Модель

Установить необходимые модули:

``` 
pandas
numpy 
faiss
matplotlib
seaborn
sklearn
```

Открыть ноутбук `model.ipynb` и выполнить все ячейки.

## Веб-сервис

Установка зависимостей:

``` 
pip install -r requirements
```

Запуск сервиса. Запускаться может долго, т.к. загружается BASE-датасет и масштабируются признаки.

``` 
flask --app server run
```

Открываем в адресной строке вот так: http://127.0.0.1:3002/recommend/3, где 3 - это идентификатор товара без `-base` суффикса.