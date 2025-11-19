import tensorflow as tf
import matplotlib.pyplot as plt

#MNIST 데이터셋 로드 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),     #28x28 이미지
    tf.keras.layers.Dense(128, activation='relu'),    #완전 연결된 신경망 층(128 뉴런)
    tf.keras.layers.Dropout(0.2),                     #과적합 방지
    tf.keras.layers.Dense(10, activation='softmax')   #0~9 숫자
])

#모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

#모델 훈련
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

#성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print("정확도: ", test_acc)

import numpy as np

#테스트 데이터 5개 예측
predictions = model.predict(x_test[:5])
print("예측 결과:", np.argmax(predictions, axis=1))
print("실제 정답:", y_test[:5])

#이미지 확인
for i in range(5):
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Pred: {np.argmax(predictions[i])}, Label: {y_test[i]}")
    plt.show()
