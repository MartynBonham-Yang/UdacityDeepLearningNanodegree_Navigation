# PROJECT: NAVIGATION - REPORT

Aim of project: code an agent which can collect bananas. Rewards are +1 for each yellow banana and -1 for each blue banana. The goal is to build an agent which can score an average of 13 points over 100 episodes. 

## First actions 

Let us quickly review the details of the state and action spaces for this project: 

* We have a **37** dimensional state space.
* There are **4** possible actions that our agent (called a "brain") can take - go forwards, go backwards, turn left and turn right. 

### First run 

Udacity provides us with a "starter code", namely, code for a brain which takes actions at random: 

```
env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
```
        
This brain (let us call him "Roland", for "random") obtains the following results over 8 episdodes: 

* Episode 1: 0.0
* Episode 2: -1.0
* Episode 3: 2.0
* Episode 4: 0.0
* Episode 5: 2.0
* Episode 6: -1.0
* Episode 7: 1.0
* Episode 8: 0.0

As we can tell, Roland does not perform very well - he needs to obtain an average score of 13 over 100 episodes, so unless these are extreme outliers 8 times consecutively, it is clear that we must make some changes. 

### Coding a better brain

When we code a brain for this project, what we are actually doing is *attempting to build an algorithm that can discover the optimal policy* - that is, we wish to find the policy `π` that results in the maximum (for this project, at least) reward for our brain. We do this by encouraging our brain to interact with it's environment and record the outcomes that it generates, and to adjust behaviour in favour of actions which generated a higher return. The process by which a brain maps states to the best action to take is called *Q-learning*.

We do this by building what is called a *Q-function* - that is, a function which calculates the reward `r` expected from taking an action `a` in a state `s`. The optimal policy `π*` can then be defined as "the policy of choosing the action that maximises the Q-function for all states in the state space". The best Q-function is the function which has the maximum reward under all such circumstances.

So how do we get started?

In this project, we are given some hints by Udacity. In particular, they state: 

> "If you're not sure where to start, here are some suggestions for how to make some progress with the project...
> In the Deep Q-Networks lesson, you applied a DQN implementation to an OpenAI Gym task. Take the time to understand this code in great detail. Tweak the various hyperparameters and settings to build your intuition for what should work well (and what doesn't!). Adapt the code from the exercise to the project, while making as few modifications as possible."

As I was indeed unsure where to start, I returned to the Deep Q-Networks lesson. In that workspace, we built an agent for the LunarLander environment, which is coded in a separate `.py` file caled `dqn_agent.py`. We should also pay attention to the fact that `dqn_agent.py` calls further on a file called `model.py`, which is used to obtain the best weights for our model. Therefore this submission also includes an `agent.py` file and a `model.py` file. 

-------

#### A note about epsilon

When our brain is learning how to behave in the world, there is always a trade-off between "following something I have done before that gets a known reward" and "do more exploration, possibly getting bad rewards, but learning new things about the environment". This is called the *exploitation-exploration dilemma*. In the course so far, we have learned about a way to balance these using the parameter epsilon: `ε`.  In my code, I have used the epsilon-greedy method outlined in the Udacity course. Epsilon-greedy methods encourage brains to learn about the environment by only following their previous highest-reward action with probability 1-ε, and choosing a random action with probability ε. 

In our code, this has a default value of 0, but we can modify this. 

#### A note about deep Q-learning

In our code, we use a method of Q-learning called deep Q-learning. This involves using a neural network to approximate an action function. Over time, the neural network weights will be updated and will eventually come to approximate the optimal policy (because our problem is a finite Markov decision process). 

The neural network which is used in this project uses 3 fully-connected layers with 64, 64, and 4 nodes respectively. [The choice of 64 is a hyperparameter. See "An explanation of model.py for my reasoning choosing 64.] The input to the first layer is 37-dimensional (because of the state space being so), and the output is 128-dimensional; after this, the next layer has 128-D input and 64-D output and then the final layer has an output of the action space size once again. 

However, as DeepMind said in their paper introducing deep Q-learning, this process can be unstable or divergent as a neural network is a nonlinear function approximator for Q. Therefore, we use something called experience replay to help mitigate this. Experience replay is inspired by biology, adn is a mechanism that relies on random sampling of previous actions instead of only paying attention to the most recent action in order to choose how to proceed. This removes correlations in the sequence of observed outcomes and helps reduce instability.


------
### An explanation of model.py

The python code file `model.py` contains the method by which we calculate the Q-function to use. This code allows us to make use of deep q-learning; that is, it lets us use a deep neural network to estimate the Q-function.  I have used the code provided by Udacity for the Deep Q-Learning lesson solution here as it works well, and have not needed to make any serious changes when adapting this for my own use. 

The hyperparameters chosen in `model.py`, namely the number of units in fc1 and fc2, are chosen as the number of nodes in the hidden layers of the model. These values are both set to 64; this is a common choice and was also the default used by Udacity. Changing these values was not necessary to make the model perform better. 

-------

When using the code provided by the UDacity Deep Q-learning exercise, we need to make some changes, but as per the instructions, 'as few changes as possible'. Therefore, the changes we have made are: 

* Replaced the state reset line of code with the equivalent from this environment (provided in the random agent model); 
* Changing the score required for the code to declare `solved` from `200.0` to `13.0`;
* Replaced the loop under `for t in range(max_t)` with the equivalent for this environment (also provided in the random agent model) - note that this is not exactly the same, because otherwise the agent would still select random actions, so we have retained `agent.step(....)` code;
* Slightly modified the code so that weights only save if we have specified `train_mode = True`, because this code is taken from the Deep Q-learning lesson which doesn't have any such parameter.
* in `agent.py`, we used hyperparameters of the following values. These are all sensible initial starting choieds and worked out very well in this project as we did not need to modify them further: 
* BATCH_SIZE = 64         
* GAMMA (future time-step discount rate) = 0.99           
* TAU (soft-update for target parameters hyperparameter) = 1e-3              
* LR (learning rate) = 5e-4            

When creating a trail brain (with `num_episodes = 200` just to see if the code runs; let's call her "Belinda") after making these tweaks, we obtained the following result: 

`Episode 200    Average Score: -0.08`

Ah. Belinda did not perform well at all and in fact it looks like she performed about as well as Roland! It turns out that we have overlooked something **very very important!!!!!**

We did not modify `action =...`, i.e we **did not tell our brain to choose an action based on our code in `agent.py`, we allowed it to continue choosing randomly!!!**

Needless to say, this was a huge mistake, and we corrected it immediately to read `action = agent.act(state, eps)` as per the Deep Q-learning code. Having made this change, my model ended up with the following brain ("Carol": 

`Episode 200    Average Score: 4.12`

So, Carol is not a successfully trained brain (yet) but is a great improvement over Belinda or Roland and proof that we could be getting somewhere. Let's try running the model for more iterations, say up to 2000 as the default code includes, and training a 4th brain ("Dorothea"):

```
Episode 100	Average Score: 1.18
Episode 200	Average Score: 4.98
Episode 300	Average Score: 7.54
Episode 400	Average Score: 9.67
Episode 500	Average Score: 12.72
Episode 510	Average Score: 13.01
Environment solved in 410 episodes!	Average Score: 13.01
```

So our code can successfully train Dorothea to the required level in just 510 episodes! 

-----

NOTE: the default code that we are given in the Deep Q-learning notebook prints here `Environment solved in {num_episodes - 100} episodes!` and I am not sure what the purpose of the `-100` is... it seems wrong to me, but this sort of code is definitely required, as this is part of the rubric. I therefore will comment here - I think that this took 510 episodes to solve, not 410, but this may be an incorrect understanding of the code. 

-----

These weights have been saved to the file `checkpoint.pth`, as per the project rubric. However, we need to do some testing, to validate these weights as useful ones, and to make sure that Dorothea can indeed perform at this level.

### Testing Dorothea, the trained brain

Testing Dorothea the trained brain for a further 10 episodes resulted in the following outcome: 
```
Episode 1	Average Score: 13.00
Episode 2	Average Score: 15.50
Episode 3	Average Score: 17.67
Episode 4	Average Score: 16.75
Episode 5	Average Score: 16.40
Episode 6	Average Score: 13.83
Episode 7	Average Score: 13.29
Episode 8	Average Score: 13.88
Episode 9	Average Score: 13.56
Episode 10	Average Score: 13.90
```

And thus we have successfully developed a brain which has a policy that returns an average reward > 13.0 when we ask it to go and collect bananas!

#### Plot of Dorothea's rewards 

This plot is taken from `Navigation.ipynb`. 
![Plot of Dorothea's training rewards over episodes](https://user-images.githubusercontent.com/57990075/167160373-322821a1-5922-4d59-96a0-d63fb8aadc79.png)

### Ideas for future work 

Some ideas which I may consider for future work on this project: 

1. Modifying further the hyperparameters, to gain better performance; 
2. Learning from pixels extension project in the Udacity course; 
3. Modifying the neural network used, perhaps to include either double or duelling Q-networks - this might improve stability in the results as we can see there is some instability. 
