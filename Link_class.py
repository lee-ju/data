# # 1. Word2Vec

# [1] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301. 3781
#
# [2] wikidocs.net/50739 "딥러닝을 이용한 자연어 처리 입문"
#
# [3] word2vec.kr/search

# ## 1-1. 패키지 호출하기

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm

# ## 1-2. 데이터 불러오기

# 최초 한 번만 실행
urllib.request.urlretrieve('https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt',
                           filename='ratings.txt')
train_data = pd.read_table('ratings.txt')
print(train_data.shape)
train_data[:5]

# ## 1-3. 전처리하기

# Null 처리
print("Null 값이 있나요? {}".format(train_data.isnull().values.any()))  # Null 확인
train_data = train_data.dropna(how='any')  # Null 제거
print(train_data.shape)

# 정규표현식으로 한글 외 문자 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data[:5]

# 불용어 처리
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘',
             '걍', '과', '도', '를', '으로', '자', '에', '와', ' 한', '한다']

# 형태소 분석기를 사용한 Tokenization
okt = Okt()
tokenized_data = []
for sentence in tqdm(train_data['document']):
    temp_x = okt.morphs(sentence, stem=True)
    temp_x = [word for word in temp_x if not word in stopwords]
    tokenized_data.append(temp_x)

# 리뷰 길이 확인
print('리뷰 최대 길이 : ', max(len(L) for L in tokenized_data))
print('리뷰 평균 길이 : ', sum(map(len, tokenized_data)) / len(tokenized_data))

plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# ## 1-4. Word2Vec 모델 구축하기

model = Word2Vec(sentences=tokenized_data,
                 vector_size=100,
                 window=5,
                 min_count=5,
                 workers=-1,
                 sg=1, # when use skip-gram. sg=0 -> CBOW
                 epochs=5000)

print("단어의 수 : {}".format(model.wv.vectors.shape[0]))
print("차원의 수 : {}".format(model.wv.vectors.shape[1]))

# ## 1-5. 모델 평가 및 저장

print(model.wv.most_similar('기생충'))

# 기생충 - 송강호
print(model.wv.most_similar(positive=['기생충'], negative=['송강호'], topn=10))

# 추격자 - 하정우
print(model.wv.most_similar(positive=['추격자'], negative=['하정우'], topn=10))

# 기생충 - 송강호 + 하정우
print(model.wv.most_similar(positive=['기생충', '하정우'], negative=['송강호'], topn=10))

# 저장하기
model.wv.save_word2vec_format('rating_w2v')

# 불러오기
from gensim.models import KeyedVectors
model_reroad = KeyedVectors.load_word2vec_format('rating_w2v')
print(model_reroad.most_similar('기생충'))

print(model_reroad.most_similar(positive=['기생충', '하정우'], negative=['송강호'], topn=10))

# # 2. VAE with flatten layer

# [1] Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv prepring arXiv:1312 6114
#
# [2] https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras

# ## 2-1. 패키지 호출하기

# + hidden=true
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

"""def sampling
@ Reparametrization Trick
@ Instead of sampling from Q(z|X), sample eps = N(0, I)
@ z = z_mean + sqrt(var) * eps
"""
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim)) # by default; mean=0 and std=1
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# ## 2-2. 데이터 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1] # 28
original_dim = image_size * image_size # 28 * 28

# ## 2-3. 전처리하기

# Preprocessing
x_train = np.reshape(x_train, [-1, original_dim]) # Flatten
x_test = np.reshape(x_test, [-1, original_dim]) # x_train ---> x_test
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Hyperparameter
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2

# ## 2-4. VAE with flatten layer 모델 구축하기

### Encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Use reparametrization trick to push the sampling out as input
# Note that 'output_shape' isn't necessary
# sampling function returns z_mean + K.exp(0.5 * z_log_var) * epsilon
z = Lambda(sampling,
           output_shape=(latent_dim, ),
           name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

### Decoder
latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

### VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# ## 2-5. 모델 학습 및 평가

if __name__ == '__main__':
    epochs = 10
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)

    #reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    mc = ModelCheckpoint('best_VAE_mnist.h5',
                         monitor='val_loss', mode='min', save_best_only=True)
    history = vae.fit(x_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(x_test, None), callbacks=[es, mc])

# ### 2-5-1. Viz: Loss

fig, loss_ax = plt.subplots()
loss_ax.plot(history.history['loss'], 'b', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')
plt.show()

# ### 2-5-2. Viz: Latent space

xmin = ymin = -4
xmax = ymax = +4

# display a 2D plot of the digit classes in the latent space
z, _, _ = encoder.predict(x_test,
                          batch_size=batch_size)
plt.figure(figsize=(12, 10))

# axes x and y ranges
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])

# subsample to reduce density of points on the plot
z = z[0::2]
y_test = y_test[0::2]
plt.scatter(z[:, 0], z[:, 1], marker="")

for i, digit in enumerate(y_test):
    axes.annotate(digit, (z[i, 0], z[i, 1]))
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()

# ### 2-5-3. Viz: Generated obj

# display a 30x30 2D manifold of digits
n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = n * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)

sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)

plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.imshow(figure, cmap='Greys_r')
plt.show()

# # 3. VAE with convolutional layer

# ## 3-1. 패키지 호출하기

from tensorflow.keras.layers import Lambda, Input, Dense,Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# ## 3-2. 데이터 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ## 3-3. 전처리하기

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3 # multi-kernel CNN
filters = 16
latent_dim = 2

# ## 3-4. VAE with conv layer 모델 구축하기

### Encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2 # filters = 32, 64
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
    
# shape info neede to build Decoder model
shape = K.int_shape(x)

# Generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Use reparametrization tricks to push the sampling out as input
# Note that 'output_shape' isn't necessary
z = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

### Decoder
latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3],
          activation='relu')(latent_inputs) # 2nd Conv2D shape; 7 * 7 * 256
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)
    filters //= 2 # 몫만 가져오기
outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_outputs')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

### VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# ## 3-5. 모델 학습 및 평가

if __name__ == '__main__':
    epochs = 10
    models = (encoder, decoder)
    data = (x_test, y_test)
    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                              K.flatten(outputs))
    #reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('best_VAE_mnist_CNN.h5', monitor='val_loss', mode='min', save_best_only=True)
    history = vae.fit(x_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(x_test, None), callbacks=[es, mc])

# ### 3-5-1. Viz: Loss

fig, loss_ax = plt.subplots()
loss_ax.plot(history.history['loss'], 'b', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')
plt.show()

# ### 3-5-2. Viz: Latent space

xmin = ymin = -4
xmax = ymax = +4

# display a 2D plot of the digit classes in the latent space
z, _, _ = encoder.predict(x_test,
                          batch_size=batch_size)
plt.figure(figsize=(12, 10))

# axes x and y ranges
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])

# subsample to reduce density of points on the plot
z = z[0::2]
y_test = y_test[0::2]

plt.scatter(z[:, 0], z[:, 1], marker="")
for i, digit in enumerate(y_test):
    axes.annotate(digit, (z[i, 0], z[i, 1]))
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()

# ### 3-5-3. Viz: Generated obj

# display a 30x30 2D manifold of digits
n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = n * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)

plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.imshow(figure, cmap='Greys_r')
plt.show()
