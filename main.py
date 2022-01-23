import os
from utils import *
from sklearn.model_selection import train_test_split

print('SETTING UP')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Importar Dados
path = 'myData'
data = import_data(path)
print(data)

# Balancer Dados para Conseguirmos uma distribuição
data = balance_data(data, display=True)

# Pegando os paths das imagens Center e a lista de ângulos de curva
img_path, steering_list = load_data(path, data)

# Train, Test, Split
x_train, x_val, y_train, y_val = train_test_split(img_path, steering_list, test_size=0.2, random_state=10)
print('Imagens para treino: ', len(x_train))
print('Imagens para validação: ', len(x_val))

# Augment Images, para diversificar o data set iremos fazer pequenas mudanças nas imagens, como flip e mudar o brilho

# Preprocessar imagens

# Batch Generator

# Criar Modelo
modelo = criar_modelo()
modelo.summary()

# Treinando o modelo
history = modelo.fit(batch_generator(x_train, y_train, 100, 1), steps_per_epoch=300, epochs=10,
                     validation_data=batch_generator(x_val, y_val, 100, 0), validation_steps=200)
modelo.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

