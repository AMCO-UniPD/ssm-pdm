---
id: j3c_paper_review
aliases: []
tags:
  - paper
---

# J3C Paper Review

I was invited to review a paper for the `J3C` conference. It is a Deep `RL` Robotics paper which is not my main area of expertise, but I think I may try to do this review (moreover the paper is from some of my colleagues at  `AMCO` so maybe I can help them with a positve review 😄).

> [!warning] Deadline
> The review has to be submitted not after June 20th, 2025.

Some important information about the paper:

- Title: *Deep Reinforcement Learning for Autonomous Navigation: Sim-to-Real Transfer on TurtleBots*
- Keywords: Autonomous Driving, Deep Reinforcement Learning, Edge Computing, Robotics
- Paper type: Special Session paper
- Corresponding author: Nicollò Turcato
- Sumbission Number: 134
- Review ID: 3117


## Paper Review

In this section I will take some notes on the paper that may help me in the review.

### Things to look at

After reading the Abstract and Introduction I realized that here the main contribution of the paper is on the experiments and on the experimental setting for the real-world experiments → this is actually a good thing because it makes the paper more understandable to me since there is probably less focus on the theoretical aspects of the algorithms (the algorithms are taken as is and I will take them for granted).

- Contribution → Not on the methodological side but more on the benchmark/evaluation side
    - A contribution may be the application of these `Deep-RL` algorithms to a real-world setting since it seems that these methods are mainly used just in simulation → I should search the literature however to have a confirmation of this.
- What I expect to see → A strong experimental setting (well described) and a well documented experimental section and may be some hints or possible solutions on how to close the `Sim2Real` gap in the future

>[!todo]
> Do a quick search on `Sim2Real` on Google Scholar and verify that there are no other works doing a similar benchmark evaluation comparison as them.

### Abstract

Deep Reinforcement Learning (Deep-RL) has emerged as a powerful paradigm for
enabling autonomous agents to learn complex behaviors in dynamic environments.
Despite its significant advancements and applications in robotics, Deep-RL
faces substantial challenges when **transitioning from simulation to real-world
deployment**, due to limited resource availability and the large amount of data
required for training. To address these issues, this paper **evaluates three
state-of-the-art continuous control Deep-RL algorithms** in the context of
**autonomous navigation tasks**. A structured experimental methodology is used
progressing from high-fidelity simulations to real-world experiments. This work
involves **more than 120 hours of real-world experiments** and shows evidence
of a possible gap between on-paper performance and real-world performance of
Deep-RL algorithms due to their different computational requirements and
assumptions.


From the abstract we can already easily understand what the paper is doing (which is already a good point for the review):

- We are in the context of `Deep-RL` applied to Robotics, in particular in autonomous navigation tasks.
- One challenge in this research field is when we transition from a simulation environment to deploying the system in the real world. This problem is so common that has its own name: `Sim2Real` gap → it's the gap between the simulated trainig environment and the real-world conditions.
- In the paper the evaluate three state-of-the-art continuous control `Deep-RL` algorithms on this autonomous navigation task and I guess that they will probably show the different performances of these methods in a simulated and a real-world environment
- They highlight the fact that they have performed 120 hourse of real-world experiment in the lab to get the results.

### Introduction

In the Introduction they introduce `RL` and its usage in robotics applications and then they introduce the problem that will be addressed in the paper. When we want to adapt this `RL` algorithms to robots and in particular when we want to pass into a real-world environment things start to get problematic, for several reasons:

- High data collection costs
- Potential hardware damage risks
- Need for extensive exploration
- Computational requirements (training time) of `RL` algorithms can be a problem in Edge Computing scenarios where we have limited resources → here they used the Edge Computing word probably to justify the `Edge Computing` keyword → let's see weather they will use it also later in the paper
- The main challenge then is the `Sim2Real` gap → because of the differences between dynamics, sensor noise and unmodelled interactions, **policies learned in simulation fail to generalize in a real-world scenario** → here I expect to see some clear experiments that show this gap.


>[!question]
> With extensive exploration are they intending the exploration of new state-action pair? Maybe yes, they are meaning that since the environment is new then the agent has to explore it a lot before finding a god policy and that takes time.

>[!question] What are update and control frequency exactly? Ask `Gemini`
> At the start of pag.2 (when they start to talk about what they will do in the paper) they talk about this "update and control frequency" which increase due to the computational requirements of the `RL` algorithms → this terms are probably some technical terms related to `Control` or `Robotics` that I have to check. Then they say that the computational constraints normally do not permit to have an high operational frequency and thus it's not possible to enable complex behaviors and achieve quick convergence through faster update rates. Maybe here are they referring to the computational speed of the hardware used in the real-world experiment?
> Moreover later in the paper they say that they focus on how control and update frequencies influence the performances.

>[!success] Update and Control Frequency
> - **Control Frequency** → How often the robot's actuators (motors, joints, etc.) receive new commands from the control system. It's the rate at which the policy outputs an action, and that action is then executed by the robot. Here the problem is that when we deploy an `RL` agent on a robot is that agent is constituted by a Deep `NN` then inference may take some time and thus the a lower number of actions is given to the robot per unit of time. Lower Control Frequency means less smooth movements
> - **Update Frequency** → This refers to how often the RL algorithm's policy parameters are updated based on new experience gathered. In simpler terms, it's how often the "brain" of the robot (the policy) learns and improves. Here the update frequency can slow down because we have to perform backpropagation to perform the gradient updates of big networks.

Then there are some initial details on the **structured experimental setting** → in the Abstract (and also in the conclusion) they highlight the fact that the experimental setting transitions from simulation to real world → here they say that they start from **realistic simulations in the Gazebo physics engine** and terminate in **real-world experiments using an experimental differential drive robot**.

>[!question]
> What do they exactly mean by starting from simulation and ending in real-world? Is there a smooth transition process from `Sim` to `Real` or there are just some experiments on `Sim` and some on `Real`. Or is there even some sort of hybrid experiments with half `Sim` and half `Real`?

### Background and Related Work

I don't want to focus too much on the Related Work section but I wil list here the three state-of-the-art model they will evaluate in the paper:

- **Deep Deterministic Policy Gradient** (`DDPG`) → Extends `Deep-RL` to continuous action spaces → critical for robotics applications. It uses an Actor Critic architecture
    - This thing seems similar to `DQN` but the action is chosen by the **actor** agent (which uses a deterministic policy) (instead of using a $ϵ$-Greedy approach to sample the action) and then the **critic** evaluates the action through Q-Learning. The actor is updated through gradient ascent with the aim of maximizing the Q function. The target networks are updated with a soft update rule.
- **Twin Delayed Deep Deterministic Policy Gradient** (`TD3`) → Addresses the overestimation bias in the value function
    - This is an improvement over `DDPG` where two Q functions are used to estimate the target value (we take the minimum of the two Q value functions) to avoid overestimatino of the Q function.
- **Soft Actor-Critic** (`SAC`) → Introduced a Maximum Entropy framework to encourage exploration and robustness in policy learning
    - `SAC` improves `DDPG` by introduction the Maximum Entropy framework where we have a stochastic policy and we want to maximize the sum of the Q function and the maximum policy entropy using the temperature $\alpha$ as the tradeoff parameter.

So → `TD3,SAC` are improving `DDPG` but they are slower in trainig because they introduce additional networks in the process and this is bad for a `Sim2Real` setting because they induce a smaller update and control frequency.

### Problem Formulation

In this section there are several details explaining the different environments used for training and evaluating the `RL` algorithms together with the three navigation tasks considered. I will just take notes on the tasks:

#### Point to Point Navigation

The robot has to move from a starting position to a target position but in the middle there is a rectangular obstacle. Clearly through the `RL` algorithm the robot should learn a policy that reaches (efficiently) the target position avoiding the middle obstacle.

>[!question] What is `atan2`?
> In the definition of the state space $s_t$ at time $t$ there is the component $e_{\theta} = \theta_t - atan2(y_g-y_t,x_g-x_t)$ (the heading error) where I don't know weather `atan2` is a typing error and they wanted to use `tan` or `arctan` or weather is some strange function I don't know about? The strange thing is that it takes two input arguments so it should not be a `tan` related function.

>[!success] `Gemini` helped me:
>The `atan2` function, also known as the **two-argument arctangent**, calculates the angle (in radians) between the positive x-axis and a point (x, y) in a Cartesian plane. It's designed to determine the angle accurately, even when the x-coordinate is zero or negative, which the standard arctangent function (atan) might struggle with

So `atan2` is essentially a fancier and improved version of `arctan`.

So after this clarification the heading error is an error on the direction of the robot with the respect to the target point to reach.

>[!question] 0.15 what?
> In the definition of the reward function for the first task, the authors say that one of the episode's terminating conditions is $d_{goal} < 0.15$, but the unit of measure is missing. What is 0.15? 0.15 m? 0.15 cm?
> The same actually holds for the position limits $[-1.2,1.2]$
> Intuitively it is probably 0.15 m and $[-1.2,1.2]$ m because the `+2` term of the reward is triggered with a threshold of 0.01 m, but for clarity maybe I have to write a comment about this on the review comment section.

>[!todo] Thing to write in the Review Comment
> For increased clarity the unit of measure should be always present, in particular in the definition of the reward function for the Point to Point Navigation task some unit of measures are missing.

Task performance metric → Success Rate → target reaching rate over 5 episodes → number of times the episodes ends with the robot reaching the goal over the total number of episodes.

>[!question] Isn't 5 episodes too low?
> I am not an expert on `RL` at all but 5 episodes seems a small sample size to me. Maybe later the time taken per experiment will be outlined and if it's very high than it may make sense to perform just 5 experiments.
> Towards the end of Section 4 they say that the low number of real-world episodes requires to have an efficient training (to not waste it in so few evaluation episodes available I guess), so probably there are some constraints which lead the authors to use just 5 evaluation episodes.

#### Path Following

In this task the robot is initialized with a random pose and then has to follow a predefined path which is defined as a sinusoidal function.

>[!question] Where is the randomness in the starting pose?
> The starting random pose is defined as $x_0 \in \{[-1,0,arctan(\pi) + \xi]^T, \xi \in [-0.1,0.1]\}$. I think that the randomness is in the $\xi$ variable but that is not specified

>[!todo] Thing to write in the Review Comment
> Better specify the source of randomness in the random pose definition in the definition of the Path Following task

>[!note]  Some doubts on the reward function
> I have some doubts on the reward function:
> - Shouldn't we insert a term for the episode termination? Like assining an high reward if terminating conditions are reached?
>   - Here one doubt can be: ok the episodes terminates when $d_{min} < 0.2$, but what, because of the random pose initialization, this condition is satisfied at the first time step? Are we sure that the robot will continue to follow the trajectory also in the future time steps? I would say that the task is completed if the path is followed until it's end, or maybe the task becomes too difficult if we pose this constraint? Actually, since the metric is the average distance covered on the x-axis if we terminate immediately because $d{min} < 0.2$ then the metric will not be very good becuase we did not cover much distance
> - Why in this case there is not a negative reward for the robot exiting the position boundaries? Maybe there is nothing regarding this because the action space they have designed make it impossible that the robot exits from the boundaries?

>[!todo] Thing to write in the Review Comment
> In the reward function definition of the Path Following task there is one thing which is not perfectly clear to me. What if, because of lucky random initialization of the robot's pose the $d_{min} < 0.2$ termination condition is satisfied after a few time steps? Is it worth considering the episode concluded in that case or maybe is it better to put an additional condition like $d_{min} < 0.2$, $x_t > 0.5$ in order to ensure that the agent has followed the path for a significant length?
> I know that, since the performance metric is the covered distance, an episode terminating after a few time steps will in any case not be considered as a very performant one but I would consider it as a bit wasted.

Performance metric for this task: Average distance metric over the x-axis, over 5 episodes.

#### Corridor Navigation

The robot is placed inside a confined rectangular ring-shaped environment, it starts from a random position inside the `corridor` parallel to one of the closed walls. The task is to navigate inside the corridors avoiding collisions with the walls. The poolicy to be learned should maximize the covered distance while avoiding obstacles and maintaining a safe navigation.

>[!question] Why the minimum laster reading is used for collision detection? Ask `Gemini`
> Here there is probably some theory related to laser readings that I don't know about.

>[!success] Laser Readings Interpretations
> The usage of these laser sensors (Lidar) is a quite common approach in robotics. The idea is that these lasers are emitting a laser beam and for each of them we measure the time taken to hit an object and being reflected back to the sensor. Since the beam is composed by light (and we know what the speed of light is) we can easily compute the distance by doing $x=c \cdot t$ where $c$ is the speed of light and $t$ is the time taken by the laser beam to come back. So obviously the smaller is the reading from the laser sensor the closer the robot is to an object, so the minimum sensor reading represent the minimum distance of the robot from an obstacle.

>[!question] What is Reward Shaping? Ask `Gemini`
> This is probably one technical thing of `RL` I have to check out.

>[!success] Reward Shaping
> Reward Shaping is used in `RL` when we have a sparse reward function so a reward function which does not trigger very often because it's conditions are difficult to satisfy. This makes it more difficult for the agent to learn the optimal policy because it may perform several actions without receiving a feedback on weather they are good or not. With Reward Shaping we assign some intermediate rewards based on the last obtained reward and the potential of the states the agent transitions between with an action. So for example with a potential function $F(\cdot)$ and a transition $(s,a)$ from state $s$ to state $s'$ reward shaping does the following → $r'(s,a,s') = r(s,a,s') + \gamma \ F(s') - F(s)$ 

>[!note] `d_{min} < 0.2`
> Here there is again the `d_{min} < 0.2` terminating condition like in the Path Following task, which I found a bit suspect.

>[!question] Why not terminating the episode in case of a collision?
> A collision with an obstacle is not a terminating condition in this reward function, differently from the Point to Point Navigation task. So why not terminating when a collision happens? Maybe this is because here we are not using the success rate as the metric but the travelled distance, if some collision happen probably the travelled distance will be small becauese the robot will stop moving?

Task performance metric: Distance travelled by the robot 5 episodes which is measured by integrating the linear velocity.

### Experimental Setup and Methodology

Ok here is what I think should be the big part of the paper.

First of all they start explaining some technical stuff on the robot, the sensors it has the procesor used on it → all things which I think are pretty standard in these kind of experiments so I guess they are correct.

The `NN` used for the policies (the actor and the critic networks) are actually very simple → feed forward networks with 2 hidden layers → probably this is also a standard thing to do in order to respect the computational constraints.

In Fig. 3 they show the control and update frequencies for 9 different network configurations (i.e. all possible configurations for 3 different values of number of hidden neurons and batch size) on each one of the 3 `RL` algorithms.

>[!todo] Thing to add in the Review Comment? (Only if I want to be a really fussy reviewer)
> The legend box in the plot in Fig. 3 is quite big, so it is not so immediate to understand weather the horizontal bars of the top left subplot are covered by the legendf box or not. Let's say that making it a little bit smaller is probably not a bad idea.

#### Training and Evaluation Strategies

Two training an evaluation strategies in the `Sim2Real` transition are considered:

- Simulation pre-training and real-world fine tuning → The robot is pre trained in the simulation environment and then it's fine tuned on the real-world one → so the networks are initialized with the last checkpoint from the simulation pre-training in the real-world environment. This is actually an interesting approach → I don't know weather it is something novel but probably not (pre-training + fine-tuning is quite a common approach these days). The authors suggest, at the end of the Section, that we should not exagerrate with the simulation pre-training because there is the risk of overfitting → the model will learn unrealistic dynamics (happening just in simulation) and will not be able to adapt to the real dynamics happening in the real-world.
- Real-world training from scratch → In this case no training is done on the simularion environment and the actor and critic networks are initialized randomly in the real-world environment.

>[!question] In the pre-training + fine-tuning scenario maybe we can afford to use bigger models?
> In this pre-training scenario, since in the simularion environment we likely have more compute resources, may it be possible to use more complex (i.e. more layers, more neurons) actor and critic networks which may then lead to better performances? I think it depends on how strict are the constraints, because having a bigger network will obviously increase the inference times in deployment and moreover we still have to fine tune it a little bit in the real-world environment.

>[!question] Why not training on simulation and just do inference in `Real`?
> This is probably not done because of the `Sim2Real` problem, policies learned on `Sim` do not generalize well on `Real`

### Results

The first good thing to highlight in the Result section is that performanes and reward statistics are **averaged over 4 random independent seeds**.

I won't insert detailed notes on the Results section but I think that it is well done and the result make sense → in particular is very interesting the final piece of result where they justify the poor performances of the `SAC` algorithm due to its low Update Frequency which makes it unable to transfer the knowledge gained in pre-training.

## Final Evaluation

I think that it makes sense to accept the paper (give an high grade → between A and B) → I still have only some doubts about that $d_{min}$ termination condition in the reward function but that may also be due to my lack of `RL` knowledge/experience.

### Final Comment

The paper evaluates three state-of-the-art Deep-RL algorithms for the task of autonomous robot navigation exploiting a structured experimental setting transitioning from simulation based training to real-world experiments, providing a quantitative and qualitative measure of the `Sim2Real` gap in the deployment of Deep-RL solutions in the field of robotics. The paper is well written and addresses a crucial problem in the field of Deep-RL applied to robotics providing some interesting insights and results.

Some technical things to adjust in the paper:

- Check the reference of the Soft Actor Critic paper → there is `????` instead of the year of publication. At then end there is `ICML-2018` so substituting `(????)` with (2018) should be enough to solve the problem.
- For increased clarity the unit of measure should be always present in the definitions of the reward functions. Only in the `+2` reward term of the reward of Point to Point Navigation (Equation 10) there is the `m` unit of measure
- In Fig. 3 reducing the legend box size may make the figure more clear since with the current size a reader may think that the horizontal bars of the plot are covered by the box.

A comment on the reward function definition in the paper:

In the reward functions definitions of the Path Following (Equation 13) task one of the episode terminating conditions is $d_{min} < 0.2$.

What if, because of lucky random initialization of the robot's pose the $d_{min} < 0.2$ termination condition is satisfied immediately or just after a few time steps? Is it worth considering the episode concluded in that case? A more detailed explanation of this aspect could make the problem definition clearer.

