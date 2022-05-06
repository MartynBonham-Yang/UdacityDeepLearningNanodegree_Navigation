# PROJECT: NAVIGATION - REPORT

Aim of project: code an agent which can collect bananas. Rewards are +1 for each yellow banana and -1 for each blue banana. The goal is to build an agent which can score an average of 13 points over 100 episodes. 

## First actions 

Let us quickly review the details of the state and action spaces for this project: 

* We have a **37** dimensional state space.
* There are **4** possible actions that our agent (called a "brain") can take - go forwards, go backwards, turn left and turn right. 

### First run 

Udacity provides us with a "starter code", namely, code for a brain which takes actions at random: 

`env_info = env.reset(train_mode=False)[brain_name] # reset the environment
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
        break`
        
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

-------

When using the code provided by the UDacity Deep Q-learning exercise, we need to make some changes, but as per the instructions, 'as few changes as possible'. Therefore, the changes I have made are: 

* Replaced the state reset line of code with the equivalent from this environment (provided in the random agent model); 
* Changing the score required for the code to declare `solved` from `200.0` to `13.0`;
* Replaced the loop under `for t in range(max_t)` with the equivalent for this environment (also provided in the random agent model) - note that this is not exactly the same, because otherwise the agent would still select random actions, so we have retained `agent.step(....)` code;
* Slightly modified the code so that weights only save if we have specified `train_mode = True`, because this code is taken from the Deep Q-learning lesson which doesn't have any such parameter.

At this point I ran into a `Error: broken pipe` error, so could not test my code immediately. 
* 
