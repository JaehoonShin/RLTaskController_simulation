# RLTaskController_simulation

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2019-0-01371, Development of brain-inspired AI with human-like intelligence) and Samsung Research Funding Center of Samsung Electronics under Project Number SRFC-TC1603-52.

#How to use

1. Install python 3.x
2. Run the main.py with arguments

# Arguments
-d : loading regdata.csv
--episodes : number of training episodes
--trials : game trials per 1 eps
--ctrl-mode : min-spe / max-spe / min-rpe / max-rpe
-n : sbj parameter index
--file-suffix : saving file suffix
--task-type : 2019/2020/2021로
2019 : controller action as nil / transition probability to [0.9,0.1] / transition probability to [0.5, 0.5] / change of goal condition (specific <-> flexible)
2020 : decaying visited state reward + controller action as nil/visited reward recover/univisted reward recover
2021 : decaying visited state reward +  controller action as nill/state-transition-probability .5<->.9 / goal state / vistited reward recover / unvisted reward recvoer 
--delta-control : visited reward decay rate

# Potential problems

# papers
1. Shin, Jae Hoon, et al. "Designing Model-Based and Model-Free Reinforcement Learning Tasks without Human Guidance." The 4th Multidisciplinary Conference on Reinforcement Learning and Decision Making (RLDM 2019). RLDM, 2019.
2. Shin, Jae Hoon, et al. "Designing Model-Based and Model-Free Reinforcement Learning Tasks without Human Guidance." 33rd Conference on Neural Information Processing Systems (NeurIPS 2019). Neural Information Processing Systems Foundation, 2019.
3. Shin, Jae Hoon, et al. "Deep Interaction between Reinforcement Learning Algorithms and Human Reinforcement Learning." 2020 한국인공지능학회 하계학술대회. (사) 한국인공지능학회, 2020
4. Shin, Jae Hoon, et al. "Deep Interaction between Reinforcement Learning Algorithms and Human Reinforcement Learning."  FROM NEUROSCIENCE TO ARTIFICIALLY INTELLIGENT SYSTEMS (NAISys), 2020.
5. Shin, Jae Hoon, et al. "In silico manipulation of human cortical computation underlying goal-directed learning." 35th Conference on Neural Information Processing Systems (NeurIPS 2021). Neural Information Processing Systems Foundation, 2021.


# references
1. Lee, Sang Wan, Shinsuke Shimojo, and John P. O’Doherty. "Neural computations underlying arbitration between model-based and model-free learning." Neuron 81.3 (2014): 687-699.
2. Yi, Sanghyun, J. Lee, and Sang Wan Lee. "Maximally separating and correlating model-based and model-free reinforcement learning." Computational and Systems Neuroscience (COSYNE). COSYNE, 2018.
