from conf import *


data = pd.read_csv(path_to_data)
data['is_russia'] = 0
data['is_russia'][data['country'] == 'Russia'] = 1


# Разибиение датасета
# train (X_train, Y_train) - 80%
# val (X_val, Y_val) - 10%
# test (X_test, Y_test) - 10%

X_train, X_tmp, Y_train, Y_tmp = train_test_split(data[['lat', 'lng']],
                                                  data['is_russia'],
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=data['is_russia'])

# test и val
X_test, X_val, Y_test, Y_val = train_test_split(X_tmp[['lat', 'lng']],
                                                Y_tmp,
                                                test_size=0.5,
                                                random_state=42,
                                                stratify=Y_tmp)

# Обучаем модель
basic_model = Sequential()
basic_model.add(Dense(units=16, activation='sigmoid', input_shape=(2,)))
basic_model.add(Dense(1, activation='hard_sigmoid'))
basic_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

basic_model.fit(np.array(X_train), np.array(Y_train), epochs=40, validation_data=(X_val, Y_val))


# оценка модели
pred = basic_model.predict(np.array(X_test))
print(roc_auc_score(Y_test, pred))
