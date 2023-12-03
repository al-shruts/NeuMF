# Recommendation system for an online store

## Installation

```sh
pip install -r requirements.txt
```

## Use model

```sh
model = NeuMF(
    ['order_dow', 'order_hour_of_day', 'days_since_prior_order', 'add_to_cart_order'],
    ['user_id', 'product_id', 'aisle_id', 'department_id']
)

model.preprocess('data/products.csv', 'data/transactions.csv')
model.load_model('model.h5')

users = [1, 2, 3]
recommendations = model.get_recommendations(users)
```

## Training models

```sh
model = NeuMF(
    ['order_dow', 'order_hour_of_day', 'days_since_prior_order', 'add_to_cart_order'],
    ['user_id', 'product_id', 'aisle_id', 'department_id']
)

model.preprocess('data/products.csv', 'data/transactions.csv')
model.train()
```