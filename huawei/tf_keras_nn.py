import pandas as pd
import numpy as np
import os
import math
import tensorflow as tf

tf.__version__

data=pd.read_csv('./2019math-huawei/train_set.csv')

y_train = data.pop('RSRP').values.reshape([-1,1])
X_train = data.copy()


def calAngle(x):
    angle = 0.0;
    x2=x[0] 
    y2=x[1]
    if  x2 == 0:
        angle = math.pi / 2.0
        if  y2 == 0 :
            angle = 0.0
        elif y2 < 0 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > 0 and y2 > 0:
        angle = math.atan(x2 / y2)
    elif  x2 > 0 and  y2 < 0 :
        angle = math.pi / 2 + math.atan(-y2 / x2)
    elif  x2 < 0 and y2 < 0 :
        angle = math.pi + math.atan(x2 / y2)
    elif  x2 < 0 and y2 > 0 :
        angle = 3.0 * math.pi / 2.0 + math.atan(y2 / -x2)
    angle1=(angle * 180 / math.pi)
    angle=min(abs(angle1-x[2]),360-abs(angle1-x[2]))
    return angle1,angle

def cal_features(df):
    data = df.copy()
    final_cols = [ 'Height', 'Azimuth', 'Frequency Band',   'RS Power', 'Cell Altitude',\
                  'Cell Building Height',  'Altitude',  'Building Height', 'detla_X', 'detla_Y', \
                  'distance', 'Downtilt', 'delta_Height','Angle_North','Angle_line','dis_3d']
    data['detla_X']=data['X']-data['Cell X']
    data['detla_Y']=data['Y']-data['Cell Y']
    data['distance']=np.sqrt((data['detla_X']*data['detla_X']+data['detla_Y']*data['detla_Y']).values)
    data['detla_Altitude']=data['Altitude']-data['Cell Altitude']
    data['detla_Building_Height']=data['Building Height']-data['Cell Building Height']
    data['Downtilt']=data['Electrical Downtilt']+data['Mechanical Downtilt']
    data[['Angle North','Angle line']] = data[['detla_X','detla_Y','Azimuth']].apply(calAngle,axis=1,result_type='expand')
    data[['Angle_North','Angle_line']]=data[['detla_X','detla_Y','Azimuth']].apply(calAngle,axis=1,result_type='expand')
    data['delta_Height']=data['Height']-(math.pi*data['Downtilt']/180).apply(math.tan)*data['distance']
    data['dis_2d'] = np.sqrt(np.power(data['X']-data['Cell X'],2)+np.power(data['Y']-data['Cell Y'],2)) 
    data['dis_3d'] = np.sqrt(np.power(data['X']-data['Cell X'],2)
                                 +np.power(data['Y']-data['Cell Y'],2)
                                 +np.power(data['Altitude']-data['Cell Altitude'],2))
    data = data[final_cols]
    print('feature done')
    return data        



def scale_X(X):
    mean = [23.487309596599854, 172.78819603503462, 2585.8694578497075,\
                11.201654547552067,502.565957359129,5.89514432956004,502.40602089647854,\
                4.088531662425362,17.01599495747214,-21.73684989363613,607.3121248431025,\
                8.670610000597328,-67.26085953571143,171.7317585825828,66.3997756962924,\
                  615.205371]
    std =  [9.45253151623637, 103.16894893842935, 5.524388461886691,\
             2.4906659761139354, 10.844284390753279, 11.509882054644466,\
             10.827945993507202, 14.460779676553404, 869.0673462667659,\
             885.7240235787392, 1082.4597349222383,3.861389458575742,\
            185.0754397949354, 105.26783711820835,49.62323152499545,\
           1090.315826 ]
    return (X-mean)/std

print('Begin calculate features...................')
# X_train = cal_features(X_train.iloc[:10000,:])
X_train = cal_features(X_train)
# uyt = cal_features(X_train.iloc[:1000,:])
# X_train = scale_X(X_train.iloc[:10000,:])
X_train = scale_X(X_train)

## train
inputs = tf.keras.Input(shape=(16,))
x = tf.keras.layers.Dense(32, activation=tf.nn.tanh)(inputs)
outputs = tf.keras.layers.Dense(1, activation=None)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse', optimizer='sgd',metrics=['mae'])
model.fit(X_train, y_train, 
#           epochs=10, batch_size=5000,validation_split=0.2)
epochs=10, batch_size=5000,validation_split=None)

## save
graph = tf.get_default_graph()
with graph.as_default():
    model.input_names = 'myInput'
    model.output_names = 'myOutput'
    tf.saved_model.simple_save(
            sess,
            './model',
            inputs={"myInput": model.input }, 
            outputs={"myOutput": model.output})



