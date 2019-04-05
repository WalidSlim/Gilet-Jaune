import numpy as np
import tensorflow as tf
from tensorflow import keras

#Organisation des données

fichier = open('groupes.txt','r',encoding='utf8')
groups=fichier.read()    
fichier.close()
groups=groups[111:]

G=groups.split('\n')
G.pop()
g1=G[0].split(';')

#création de X (input)
g=G[0].split(';')
g.pop(0)
l=[float(i) for i in g]
X=np.array(l,np.float32)
X=np.reshape(X,(1,10))
for i in range(1,len(G)):
    g=G[i].split(';')
    g.pop(0)
    l1=[float(k) for k in g]
    l2=np.array([l1],np.float32)
    X=np.concatenate((X,l2))

#Creation de Y (output)
Y=np.array([int('B' in i) for i in G])

train_X=X[:700]
test_X=X[700:]
train_Y=Y[:700]
test_Y=Y[700:]

model=keras.models.Sequential()
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dense(1,activation='sigmoid'))
sgd=keras.optimizers.SGD(lr=0.001)
#relu + opti 0.025
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_X,train_Y,epochs=10,validation_data=(test_X,test_Y))




fichier = open('groupes.txt','r',encoding='utf8')
groups=fichier.read()    
fichier.close()
groups=groups[111:]

G=groups.split('\n')
G.pop()
g1=G[0].split(';')

#création de X (input)
g=G[0].split(';')
g.pop(0)
l=[float(i) for i in g]
X=np.array(l,np.float32)
X=np.reshape(X,(1,10))
for i in range(1,len(G)):
    g=G[i].split(';')
    g.pop(0)
    l1=[float(k) for k in g]
    l2=np.array([l1],np.float32)
    X=np.concatenate((X,l2))

#Creation de Y (output)
Y=np.array([int('B' in i) for i in G])
model.evaluate(X,Y)


#group4
#sans colonnes
fichier=open('groupes4.txt','r')
group2=fichier.read()
fichier.close()
group2=group2[111:]
G=group2.split('\n')
G.pop()
g1=G[0].split(';')
g=G[0].split(';')
y=[]
y.append(g.pop(0))
l=[float(i) for i in g]
X=np.array(l,np.float32)
X=np.reshape(X,(1,10))
for i in range(1,len(G)):
    g=G[i].split(';')
    y.append(g.pop(0))
    l1=[float(k) for k in g]
    l2=np.array([l1],np.float32)
    X=np.concatenate((X,l2))
Y=np.array([int('B' in i)*0.5+int('C' in i) for i in G])
Y=keras.utils.to_categorical(Y,num_classes=3)
train_X=X[:100]
test_X=X[900:]
train_Y=Y[:100]
test_Y=Y[900:]

model=keras.models.Sequential()
#model.add(keras.layers.Dense(16,activation='relu',input_shape=(10,)))
#model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(3,activation='sigmoid'))
sgd=keras.optimizers.SGD(lr=0.05,momentum=0.5,nesterov=True)
#cchange to 0.06 et test softmax
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_X,train_Y,epochs=6,validation_data=(test_X,test_Y))

model.evaluate(X[100:900],Y[100:900])
m=model.predict(X)

