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

### Modifying the project code

In this project, we are given some hints by Udacity. In particular, they state: 

> "If you're not sure where to start, here are some suggestions for how to make some progress with the project...
> In the Deep Q-Networks lesson, you applied a DQN implementation to an OpenAI Gym task. Take the time to understand this code in great detail. Tweak the various hyperparameters and settings to build your intuition for what should work well (and what doesn't!). Adapt the code from the exercise to the project, while making as few modifications as possible."

As I was unsure where to start, I returned to the Deep Q-Networks lesson. In that workspace, we built an agent for the LunarLander environment, which is coded in a separate .py file caled Agent. We should also pay attention to the fact that Agent.py calls further on a file called model.py, which is used to obtain the best weights for our model.

To begin with, I used 
