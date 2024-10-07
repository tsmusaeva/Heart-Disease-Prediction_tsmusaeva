import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Определение числовых и категориальных признаков
numeric_columns = ['age', 'chest', 'resting_blood_pressure', 'serum_cholestoral',
                   'maximum_heart_rate_achieved', 'oldpeak']
categorical_columns = ['sex', 'fasting_blood_sugar', 'resting_electrocardiographic_results',
                       'exercise_induced_angina', 'slope', 'number_of_major_vessels', 'thal']

# Загрузка модели
loaded_model = joblib.load('best_random_forest_model.pkl')

# Загрузка препроцессора
preprocessor = joblib.load('preprocessor.pkl')

# Загрузка новых данных из CSV файла
new_data_file_path = '../test.csv'
new_data = pd.read_csv(new_data_file_path)

# Удаление столбца ID, если он есть
if 'ID' in new_data.columns:
    new_data.drop('ID', axis=1, inplace=True)
# Проверка наличия необходимых столбцов
missing_columns = set(numeric_columns + categorical_columns) - set(new_data.columns)
if missing_columns:
    print(f"Отсутствуют следующие столбцы в новых данных: {missing_columns}")
else:
    print("New Data:\n", new_data)

    preprocessor.transform(new_data)
    print("Transformed Data:\n", new_data)
    predictions = loaded_model.predict(new_data)
    print("Predictions:", predictions)
    #Вывод вероятностей для каждого класса
    probabilities = loaded_model.predict_proba(new_data)
    print("Probabilities:\n", probabilities)
