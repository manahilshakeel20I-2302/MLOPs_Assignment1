import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load and preprocess data
def load_and_preprocess_data(filepath):
<<<<<<< HEAD
    filepath = "C:/Users/DELL/Documents/GitHub/MLOPs_Assignment1/House_Prices.csv"

=======
    filepath= "C:/Users/DELL/Downloads/archive (1)/House_Prices.csv"
>>>>>>> 64496404de2fe93277205807f33764273537d179
    data = pd.read_csv(filepath)

    X = data.drop(columns=['Price'])  # 'Price' is the target column
    y = data['Price']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X_processed = preprocessor.fit_transform(X)
<<<<<<< HEAD
    return X_processed, y, preprocessor
=======
    return X_processed, y, preprocessor

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"Model score: {model.score(X_test, y_test)}")
    return model

# Save model and preprocessor
def save_model(model, preprocessor):
    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

# Load model and preprocessor
def load_model():
    model = joblib.load('house_price_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

# Predict house price
def predict_price(features, model, preprocessor):
    features = np.array(features).reshape(1, -1)
    processed_features = preprocessor.transform(features)
    prediction = model.predict(processed_features)
    return prediction[0]

# Main script (Train and save the model)
if __name__ == "__main__":
    X, y, preprocessor = load_and_preprocess_data('house_prices.csv')
    model = train_model(X, y)
    save_model(model, preprocessor)
>>>>>>> 64496404de2fe93277205807f33764273537d179
