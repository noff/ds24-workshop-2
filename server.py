import faiss
from flask import Flask
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

total_steps = 8
print(f'1/{total_steps}: создаем приложение')

app = Flask(__name__)

print(f'2/{total_steps}: настраиваем RANDOM_STATE')
RANDOM_STATE = 12345
np.random.seed(RANDOM_STATE)

# Подготовка датасета с характеристиками товаров
print(f'3/{total_steps}: читаем датасет')
df_base = pd.read_csv('data/base.csv',
                      index_col='Id',
                      # nrows=1000,
                      usecols=['Id', '0', '1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22', '23', '24', '26', '27', '28', '29', '30', '31', '32', '34', '35', '36', '37', '38', '39', '41', '43', '44', '45', '46', '47', '48', '49', '50', '51', '53', '55', '56', '57', '58', '62', '63', '64', '67', '68', '69', '70', '71']
                      ).astype('float32')

print(f'4/{total_steps}: масштабируем датасет')
transformer = RobustScaler().fit(df_base.values)
df_base = pd.DataFrame(index=df_base.index, data=transformer.transform(df_base.values))

# Идентификаторы существующих товаров
print(f'5/{total_steps}: формируем списки товаров')
product_ids = list(df_base.index)

# Словарь для конвертации порядкового номера товара в идентификатор
print(f'6/{total_steps}: формируем словарь идентификаторов товаров')
base_index = {k: v for k, v in enumerate(df_base.index.to_list())}

# Загружаем модель
print(f'7/{total_steps}: загружаем модель')
model = faiss.read_index('./model.index')

print(f'8/{total_steps}: поехали')
@app.route("/")
def home():
    response = app.response_class(
        response="Используйте адреса типа /recommend/XXX,\nгде XXX - число от 0 до 4744766. \nЭто идентификаторы товаров. Но не все товар могут быть в базе, поэтому иногда API будет возвращать ошибку 404. \nТовары для примера: 0, 1, 2, 557, 886, 4744766",
        status=200,
        mimetype='text/plain'
    )
    return response

@app.route("/recommend/<int:product_id>")
def recommend(product_id):

    # Готовим идентификатор товара
    product_code = f'{product_id}-base'

    # Если нет товара в базе, возвращаем ошибку
    if product_code not in product_ids:
        return {
            'status': 'fail',
            'data': {
                'message': f'Product {product_id} is not found'
            }
        }, 404

    # Готовим предсказание
    r, idx = model.search(np.ascontiguousarray([df_base.loc[product_code, :].tolist()]).astype('float32'), 6)

    # Возвращаем ответ
    return {
        'status': 'success',
        'data': {
            'product_ids': [base_index[i] for i in idx.tolist()[0] if base_index[i] != product_code]
        }
    }
