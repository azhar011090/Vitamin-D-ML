
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Генерация синтетического датасета
np.random.seed(42)
n_samples = 500
df = pd.DataFrame({
    'Age': np.random.randint(18, 80, size=n_samples),
    'Sex': np.random.choice([0, 1], size=n_samples),
    'BMI': np.round(np.random.normal(25, 5, size=n_samples), 1),
    'Skin_Tone': np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.4, 0.1]),
    'Region_Sun': np.random.choice([1, 2, 3], size=n_samples, p=[0.3, 0.5, 0.2]),
    'Sun_Exposure': np.clip(np.round(np.random.exponential(1, size=n_samples) * 2, 1), 0, 6),
    'Supplement_Intake': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    'Season': np.random.choice([1, 2, 3, 4], size=n_samples)
})

df['Vitamin_D'] = (
    20
    + (df['Sun_Exposure'] * 2)
    + (df['Region_Sun'] * 2)
    + (df['Supplement_Intake'] * 10)
    - (df['Skin_Tone'] * 2)
    - (df['Season'] == 1) * 5
    + np.random.normal(0, 5, size=n_samples)
)
df['Vitamin_D'] = np.clip(df['Vitamin_D'], 5, 80)

# Признаки и целевая переменная
X = df.drop(columns='Vitamin_D')
y = df['Vitamin_D']

# OneHotEncoder
categorical_features = ['Skin_Tone', 'Region_Sun', 'Season', 'Sex']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), categorical_features)],
    remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(X)

# Обучение модели
model_lin = LinearRegression()
model_lin.fit(X_encoded, y)

# Функция для предсказания
def predict_vitamin_d(age, sex, bmi, skin_tone, region_sun, sun_exposure, supplement_intake, season):
    input_df = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Skin_Tone': skin_tone,
        'Region_Sun': region_sun,
        'Sun_Exposure': sun_exposure,
        'Supplement_Intake': supplement_intake,
        'Season': season
    }])
    input_encoded = preprocessor.transform(input_df)
    prediction = model_lin.predict(input_encoded)
    return round(prediction[0], 1)

# Streamlit UI
st.title('💊 Прогноз уровня витамина D')

st.write('Введите данные клиента:')

age = st.number_input('Возраст', min_value=0, max_value=120, value=30)
sex = st.selectbox('Пол', options=[('Женщина', 0), ('Мужчина', 1)], format_func=lambda x: x[0])[1]
bmi = st.number_input('Индекс массы тела (BMI)', min_value=10.0, max_value=60.0, value=25.0)
skin_tone = st.selectbox('Цвет кожи', options=[('Светлый', 1), ('Средний', 2), ('Тёмный', 3)], format_func=lambda x: x[0])[1]
region_sun = st.selectbox('Солнечность региона', options=[('Низкая', 1), ('Средняя', 2), ('Высокая', 3)], format_func=lambda x: x[0])[1]
sun_exposure = st.number_input('Время на солнце в день (часы)', min_value=0.0, max_value=10.0, value=1.0)
supplement_intake = st.selectbox('Принимает витамин D?', options=[('Нет', 0), ('Да', 1)], format_func=lambda x: x[0])[1]
season = st.selectbox('Сезон', options=[('Зима', 1), ('Весна', 2), ('Лето', 3), ('Осень', 4)], format_func=lambda x: x[0])[1]

if st.button('🔮 Прогнозировать'):
    result = predict_vitamin_d(age, sex, bmi, skin_tone, region_sun, sun_exposure, supplement_intake, season)
    st.success(f"💡 Прогнозируемый уровень витамина D: {result} нг/мл")
