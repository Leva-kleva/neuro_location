from conf import *


data = pd.read_csv(path_to_data)
data['is_russia'] = 0
data['is_russia'][data['country'] == 'Russia'] = 1

# Нормализация данных
from sklearn import preprocessing

data['lat_norm'] = preprocessing.StandardScaler().fit_transform(data[['lat']])
data['lng_norm'] = preprocessing.StandardScaler().fit_transform(data[['lng']])


# Разибиение датасета
# train (X_train, Y_train) - 70%
# val (X_val, Y_val) - 10%
# test (X_test, Y_test) - 20%

X_train, X_tmp, Y_train, Y_tmp = train_test_split(data[['lat_norm', 'lng_norm']],
                                                  data['is_russia'],
                                                  test_size=0.3,
                                                  random_state=42,
                                                  stratify=data['is_russia'])

# test и val
X_test, X_val, Y_test, Y_val = train_test_split(X_tmp[['lat_norm', 'lng_norm']],
                                                Y_tmp,
                                                test_size=0.33,
                                                random_state=42,
                                                stratify=Y_tmp)

# Обучаем модель
basic_model = Sequential()
basic_model.add(Dense(units=16, activation='sigmoid', input_shape=(2,)))
basic_model.add(Dense(1, activation='hard_sigmoid'))
sgd = optimizers.SGD(learning_rate=0.5, momentum=0.9, nesterov=True)
basic_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[tf.keras.metrics.AUC()])

basic_model.fit(np.array(X_train), np.array(Y_train), epochs=40, validation_data=(X_val, Y_val))


# оценка модели
pred = basic_model.predict(np.array(X_test))
print(roc_auc_score(Y_test, pred))
