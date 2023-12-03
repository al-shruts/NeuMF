import numpy as np
import pandas as pd

from typing import Optional
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, Input, Dense, Concatenate, Flatten, Multiply, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class NeuMF:
    """
    Neural Matrix Factorization (NeuMF) Recommender System.

    Attributes:
        input_data (pd.DataFrame): Processed input data.
        target_data (pd.Series): Target variable for training.
        history (tf.keras.callbacks.History): Training history.
        model (tf.keras.models.Model): NeuMF model.
        label_encoders (dict): Dictionary storing label encoders for categorical features.
        nums_categorical_features (dict): Dictionary storing the number of categories for each categorical feature.
        numeric_features (list[str]): List of names of numerical features.
        categorical_features (list[str]): List of names of categorical features.
    """

    def __init__(self, numeric_features: list[str], categorical_features: list[str]):
        """
        Initializes NeuMF with given numerical and categorical features.

        Args:
            numeric_features (list[str]): List of names of numerical features.
            categorical_features (list[str]): List of names of categorical features.
        """

        self.input_data = None
        self.target_data = None

        self.history = None
        self.model = None
        self.label_encoders = {}
        self.nums_categorical_features = {}

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def preprocess(self, products_path: str, transactions_path: str):
        """
        Preprocesses raw data for training the NeuMF model.

        Args:
            products_path (str): Path to the products data file.
            transactions_path (str): Path to the transactions data file.
        """

        df = pd.merge(pd.read_csv(products_path), pd.read_csv(transactions_path), on='product_id', how='left')
        df.fillna(0, inplace=True)

        df = df.groupby(['user_id', 'product_id']).agg({
            'aisle_id': 'first',
            'department_id': 'first',
            'order_id': 'first',
            'order_number': 'mean',
            'order_dow': 'mean',
            'order_hour_of_day': 'mean',
            'days_since_prior_order': 'mean',
            'add_to_cart_order': 'mean',
            'reordered': 'last',
            'product_name': 'first',
            'aisle': 'first',
            'department': 'first'
        }).reset_index()

        # Normalization of numerical features
        scaler = MinMaxScaler()
        df[self.numeric_features] = scaler.fit_transform(df[self.numeric_features])

        # Coding of categorical features
        for feature in self.categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            self.label_encoders[feature] = le
            self.nums_categorical_features[feature] = len(le.classes_)

        self.input_data = df[self.numeric_features + self.categorical_features]

        self.target_data = df['reordered']

    def build(self, latent_dim: int = 10, layers: list[int] = [20, 10],
              reg_layers: list[int] = [0, 0], reg_mf: list[int] = [0, 0],
              dropout_mlp: float = 1e-3, learning_rate: float = 1e-5):

        """
        Builds the NeuMF model.

        Args:
            latent_dim (int): Dimension of the latent space.
            layers (list[int]): List of layer sizes for the MLP component.
            reg_layers (list[int]): List of regularization strengths for MLP layers.
            reg_mf (list[int]): List of regularization strengths for MF layers.
            dropout_mlp (float): Dropout rate for MLP layers.
            learning_rate (float): Learning rate for model training.
        """

        # Inputs
        inputs = {}
        for feature in self.categorical_features:
            inputs[feature] = Input(shape=(1,), dtype='int32', name=f'{feature}_input')
        inputs['numeric'] = Input(shape=(len(self.numeric_features),), dtype='float32', name='numeric_inputs')

        # GMF Embeddings
        MF_User = Flatten()(Embedding(input_dim=self.nums_categorical_features[self.categorical_features[0]],
                                      output_dim=latent_dim,
                                      name='mf_user',
                                      embeddings_initializer='random_normal',
                                      embeddings_regularizer=l2(reg_mf[0]),
                                      input_length=1)(inputs[self.categorical_features[0]]))
        MF_Item = Flatten()(Embedding(input_dim=self.nums_categorical_features[self.categorical_features[1]],
                                      output_dim=latent_dim,
                                      name='mf_item',
                                      embeddings_initializer='random_normal',
                                      embeddings_regularizer=l2(reg_mf[1]),
                                      input_length=1)(inputs[self.categorical_features[1]]))

        # GMF
        GMF_vector = Multiply()([MF_User, MF_Item])

        # MLP Embeddings
        MLP = {}
        for feature in self.categorical_features:
            MLP[feature] = Flatten()(Embedding(input_dim=self.nums_categorical_features[feature],
                                               output_dim=int(layers[0] / 2),
                                               name=f'mlp_{feature}',
                                               embeddings_initializer='random_normal',
                                               embeddings_regularizer=l2(reg_layers[0]),
                                               input_length=1)(inputs[feature]))

        # MLP
        MLP_vector = Concatenate()(list(MLP.values()) + [inputs['numeric']])

        for idx, layer in enumerate(layers):
            MLP_vector = Dense(layer, activation='relu', kernel_regularizer=l2(reg_layers[idx]))(MLP_vector)
            MLP_vector = Dropout(dropout_mlp)(MLP_vector)  # Add dropout for regularization (optional)

        # Combine GMF + MLP
        predict_vector = Concatenate()([GMF_vector, MLP_vector])

        # Final prediction layer
        prediction = Dense(1, activation='sigmoid',
                           kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

        self.model = Model(inputs=list(inputs.values()), outputs=prediction)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')

    def load_model(self, model_path: str):
        """
        Loads a pre-trained NeuMF model.

        Args:
            model_path (str): Path to the saved model file.
        """

        self.model = load_model(model_path)

    def fit(self, epochs: int = 5, batch_size: int = 32, validation_split: float = 1e-2, verbose: str = 'auto',
            min_delta: float = 1e-0, patience: int = 0, model_checkpoint_path: str = 'model.h5'):
        """
        Trains the NeuMF model.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches during training.
            validation_split (float): Fraction of the training data to be used as validation set.
            verbose (str): Verbosity mode (auto, 0, 1, 2).
            min_delta (float): Minimum change to qualify as an improvement in the validation loss.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            model_checkpoint_path (str): Path to save the best model.
        """

        # Define the EarlyStopping and ModelCheckpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience)
        model_checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

        # Train the model using the callbacks
        self.history = self.model.fit(
            [self.input_data[feature].values for feature in self.categorical_features] + [
                self.input_data[self.numeric_features].values],
            self.target_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint],
            verbose=verbose
        )

    def predict(self, users: list[int], batch_size: Optional[int] = None):
        """
        Generates predictions for a list of users.

        Args:
            users (list[int]): List of user IDs.
            batch_size (Optional[int]): Batch size for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """

        # Extract unique items from the categorical feature at index 1 in the input data
        item = self.input_data[self.categorical_features[1]].unique()

        # Duplicate the users and items to create pairs for prediction
        user = np.tile(np.array(users)[:, np.newaxis], (1, len(item))).ravel()
        item = np.tile(item, len(users))

        # Extract item information based on the duplicated items
        item_info = self.input_data[self.input_data[self.categorical_features[1]].isin(item)].drop_duplicates(
            subset=self.categorical_features[1])

        # Create lists of duplicated categorical features for each item
        item_data = [np.tile(np.array(item_info[feature]), len(users)) for feature in self.categorical_features[2:]]

        # Duplicate numeric features for each user
        numeric_data = np.array(item_info[self.numeric_features].values)
        numeric_data = np.tile(numeric_data, (len(users), 1))

        # Use the model to predict based on the user-item pairs and additional features
        return self.model.predict([user, item] + item_data + [numeric_data], batch_size=batch_size)

    def get_recommendations(self, users: list[int], K: int = 10, batch_size: Optional[int] = None):
        """
        Gets top K recommendations for a list of users.

        Args:
            users (list[int]): List of user IDs.
            K (int): Number of recommendations to retrieve.
            batch_size (Optional[int]): Batch size for prediction.

        Returns:
            list: List of recommended items for each user.
        """

        # Get predictions for the given users using the predict method
        predictions = self.predict(users, batch_size)

        # Reshape the predictions to have a shape of (number of users, number of items)
        predictions = predictions.reshape(len(users), len(predictions) // len(users))

        # Get the indices of the top K recommendations for each user
        recommendations = (-predictions).argsort(axis=1)[:, :K]

        # Inverse transform the indices to get the original categorical values of the recommended items
        return [self.label_encoders[self.categorical_features[1]].inverse_transform(recommend) for recommend in
                recommendations]
