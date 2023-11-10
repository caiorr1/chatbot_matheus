# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Carregar o conjunto de dados
data = pd.read_csv('diabetes.csv')

# Realizar pré-processamento dos dados
# Aqui, não há tratamento de valores ausentes para simplicidade, mas você deve considerar fazê-lo em um caso real
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escolher um modelo de classificação (Random Forest neste exemplo)
model = RandomForestClassifier()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy}')

# Salvar o modelo treinado
joblib.dump(model, 'diabetes_model.pkl')

