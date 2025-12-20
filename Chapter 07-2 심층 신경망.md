## 코드 전문
```python
import keras
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1, 28*28)
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state = 42
)

print(train_scaled.shape, train_target.shape)

inputs = keras.layers.Input(shape=(784,))

### 각 층 객체별 instance를 만들어 전달하는 방법
dense1 = keras.layers.Dense(100, activation = 'sigmoid') # 밀집 층
dense2 = keras.layers.Dense(10, activation = 'softmax') # 출력 층
model = keras.Sequential([inputs, dense1, dense2]) # 입력 층을 맨 앞에 두고, 출력층을 가장 마지막에 두어야 함

### 생성자에 한 번에 전달하는 방법
model = keras.Sequential([
    keras.layers.Input(shape = (784, )),
    keras.layers.Dense(100, activation = 'sigmoid', name = 'hidden_layer'),
    keras.layers.Dense(10, activation = 'softmax', name = 'output_layer')
], name = '패션 MNIST 모델')

### add 메서드를 사용하는 방법
model = keras.Sequential()
model.add(keras.layers.Input(shape = (784, )))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'sigmoid', name = 'hidden_layer'))
model.add(keras.layers.Dense(10, activation = 'softmax', name = 'output_layer'))


model = keras.Sequential()
model.add(keras.layers.Input(shape = (28,28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
train_scaled  = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state = 42
)

model = keras.Sequential()
model.add(keras.layers.Input(shape = (28,28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_scaled, train_target, epochs = 5)

model.evaluate(val_scaled, val_target)
```
