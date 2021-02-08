### Export to python
```
jupytext --set-formats ipynb,py:percent Pegasus.ipynb
jupytext --to notebook notebook.py
```

conda install jupyterlab ipykernel numpy scipy pytorch cudatoolkit=11.0 matplotlib tqdm umap-learn pandas  -c conda-forge

## Pegasus
https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
https://syncedreview.com/2019/06/06/going-beyond-gan-new-deepmind-vae-model-generates-high-fidelity-human-faces/
https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed

https://towardsdatascience.com/introduction-to-resnets-c0a830a288a4
https://arxiv.org/pdf/1512.03385.pdf
https://www.youtube.com/watch?v=DkNIBBBvcPs&feature=emb_rel_pause
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py


https://orybkin.github.io/sigma-vae/
https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/#kingma2013auto
https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/

https://arxiv.org/pdf/1905.02417.pdf
https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b


## RL

https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c
https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/

https://www.freecodecamp.org/news/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f/
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
https://arxiv.org/abs/1602.01783

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

Using atari as benchmark: https://arxiv.org/abs/1705.06936
https://deepsense.ai/solving-atari-games-with-distributed-reinforcement-learning/
lowkey ram: https://deepsense.ai/playing-atari-on-ram-with-deep-q-learning/
Ram or others: https://jair.org/index.php/jair/article/view/10819/25823

https://medium.com/@SmartLabAI/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc

PPO the boi: https://openai.com/blog/baselines-acktr-a2c/
https://arxiv.org/pdf/1707.06347.pdf
https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26

https://towardsdatascience.com/proximal-policy-optimization-ppo-with-sonic-the-hedgehog-2-and-3-c9c21dbed5e


https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

cnn sizes: https://github.com/openai/baselines/blob/d3fed181b57f61013698521f4e940594364253e9/baselines/ppo1/cnn_policy.py