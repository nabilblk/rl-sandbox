## Prérequis : 

* Gym 
* Tensorflow
* Keras
* Keras-rl 

### Installation de Gym : 

```
sudo apt-get install -y python3-numpy python3-dev python3-pip cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
cd ~
git clone https://github.com/openai/gym.git
cd gym
sudo pip3 install -e '.[all]'
```

## Experiment 5
 
 * `Taxi-v2` : Example .  
 * Random Steps .   
 * Simple Learning Formula :
  
![](/assets/learning_formula.png)
 
## Experiment 6

 * More complex environement `Lunarlander-v2`
 * Using keras-rl for it's simplicity .

> DL model : 
 
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 8)                 0         
_________________________________________________________________
dense_1 (Dense)              (None, 16)                    144       
_________________________________________________________________
activation_1 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 16)                272       
_________________________________________________________________
activation_2 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 16)                272       
_________________________________________________________________
activation_3 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 4)                 68        
_________________________________________________________________
activation_4 (Activation)    (None, 4)                 0         
=================================================================
Total params: 756
Trainable params: 756
Non-trainable params: 0
_________________________________________________________________
```


## Lectures : 
 * https://en.wikipedia.org/wiki/Q-learning