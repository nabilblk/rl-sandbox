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

## Experiment 1
 
 * `Taxi-v2` : Example .  
 * Random Steps .   
 * Simple Learning Formula :
  
![](/assets/learning_formula.png)
 
## Experiment 2

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
 * https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
 * https://en.wikipedia.org/wiki/Q-learning
 * https://hackernoon.com/the-3-tricks-that-made-alphago-zero-work-f3d47b6686ef
 * https://github.com/deepmind/sonnet
 * https://github.com/deepmind/lab
 * https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
 * https://ai.intel.com/demystifying-deep-reinforcement-learning/
 * http://awjuliani.github.io/exploration/index.html
 * https://www.chess.com/news/view/google-s-alphazero-destroys-stockfish-in-100-game-match
 * https://github.com/wagonhelm/Reinforcement-Learning-Introduction/blob/master/Reinforcement%20Learning%20Introduction.ipynb
 * https://raw.githubusercontent.com/ageron/tiny-dqn/master/tiny_dqn.py
 * https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
