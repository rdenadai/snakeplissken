## Snake Plissken

It's a small snake game build using pygame which plays by itself using Dueling DQN implemented with Pytorch.

If you try to run the game from zero it took more than 1 million iterations to get to this place (see image bellow).

My hardware to train the Agent is:
 - Core i7 (8 cores)
 - 16 Mb
 - NVIDIA Geforce 1060 with 6 Gb

### Running

> Run **main.py** if you want to train and watch the Agent!

> Run **play.py** if you just want to watch the Agent!

> Run **train.py** if you already run main.py and have some memory... the Agent will train on stored memory (this is not the common pratice).


### Model

The model was based on the original Dueling DQN presented in the paper. I used RMSProp and MSE as proposed in the same paper as the Optmizer and Loss.

![Dueling DQN](https://github.com/rdenadai/snakeplissken/blob/master/img/dueling.png)

To know more: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)


### Example of playing

| ![Agent Playing](https://github.com/rdenadai/snakeplissken/blob/master/img/snake.gif)  | ![Agent Playing](https://github.com/rdenadai/snakeplissken/blob/master/img/snake2.gif)  | ![Agent Playing](https://github.com/rdenadai/snakeplissken/blob/master/img/snake3.gif)  |
