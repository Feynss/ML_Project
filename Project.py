# Подключаем библиотеки и загружаем данные.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('./column_2C_weka.csv')

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)

# Смотрим на размеры наших данных.
print(df.shape)

# Смотрим на данные в целом.
print(df.info())

# Смотрим на значения признака class с помощью столбчатой диаграммы.
df['class'].value_counts().plot(kind='bar')
plt.show()

# Мини-статистика по данным.
print(df.describe())

# Строим матрицу корреляций с тепловой диаграммой.
corr_matrix = df.drop([], axis=1).corr()

sns.heatmap(corr_matrix)
plt.show()

# Строим гистограммы для числовых признаков.
hist_columns = ['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius',
                'degree_spondylolisthesis']

for column_name in hist_columns:
    df[column_name].plot(kind='hist', title=column_name)
    plt.show()

# Не забываем построить с ними и dist plot.
for column_name in hist_columns:
    sns.distplot(df[column_name], label=column_name)
    plt.show()

# Строим box-plot-ы для числовых признаков и нашего признака class.
for column in hist_columns:
    sns.boxplot(y="class", x=column, data=df, orient="h")
    plt.show()

# А теперь построим модель.

# Факторизуем и отделяем наш столбец class.
df['class'] = pd.factorize(df['class'])[0]
y = df['class']

# Избавляемся от столбца, который предсказываем.
df.drop(['class'], axis=1, inplace=True)

# Разделим нашу выборку на две части - для обучения и для тестирования.
# Для этого воспользуемся методом train_test_split из model_selection из sklearn.
X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3, random_state=17)

# Строим нашу модель.
knn = KNeighborsClassifier(n_neighbors=10)

# Обучаем модель.
knn.fit(X_train, y_train)

# Делаем предсказание на оставшиеся данные.
knn_pred = knn.predict(X_holdout)

# Воспользуемся методом accuracy_score из metrics из sklearn.
accuracy = accuracy_score(y_holdout, knn_pred)
print(accuracy)
