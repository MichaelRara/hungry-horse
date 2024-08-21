# hungry_horse

This repository contains implemetation of several basic RL algorithms such as SARSA, n-step SARSA and Q-Learning. 
The problem is to find a way through a chessboard from bottom-left state to top right state. Only possibilities of movement corresponds to actions of chess horse (Move only in a shape of letter L).

User can define his own chessboard, rewards of step, initial values of states and standard parameters of RL algorithms mentioned above. It is also possible to manually change reward of steps on selected states which are labeled as "deadly states" and marked by red titles.

Detected solution is saved as a set of jpg images into working directory. Bear in mind the algorithm can crash and get stuck in a loop jumping from one state to previous and back again. This situation occurs if parameters of RL algorithm are set poorly. In such a case infinity amount of jpg files will be stored into your pc. I strognly suggest to use debuger for this vizualization.

# Chessboard example
Every step has reward -1. If a horse make a step on the red state the reward is -100. The goal is to find a way from bottom left corner into top right corner with lowest sum of rewards.
Actual position of a horse is vizualized by orange color. Next position is vizualized by purple color.



![Chessboard_0](https://github.com/user-attachments/assets/74479bf1-c96a-4248-8828-ad0a6cecbb61)
![Chessboard_1](https://github.com/user-attachments/assets/ec2e28a9-dc0b-423b-a510-ca2ce9829f67)
![Chessboard_2](https://github.com/user-attachments/assets/19f032c1-f9da-4947-9f42-bebd07c218e7)
![Chessboard_3](https://github.com/user-attachments/assets/5e407bbd-5bf9-4c00-bba9-7876b9864768)
![Chessboard_4](https://github.com/user-attachments/assets/6a5877a4-324d-4e41-aee7-1fd4025cf485)
![Chessboard_5](https://github.com/user-attachments/assets/2d803031-20e1-4f8c-8131-ba1bfeb5067b)
![Chessboard_6](https://github.com/user-attachments/assets/57757c42-9251-4977-82b4-407e3ebd5b72)
![Chessboard_7](https://github.com/user-attachments/assets/8c1e3974-dfb2-436d-ab75-6cccbe50a233)
![Chessboard_8](https://github.com/user-attachments/assets/611e6aef-d40f-4602-8a89-1624e51bc5c9)
![Chessboard_9](https://github.com/user-attachments/assets/ddf71aa2-26e4-4b15-b4bd-f92323eadcf1)
![Chessboard_10](https://github.com/user-attachments/assets/d3c1731a-d130-4d2f-8ad7-22acacfdd937)
![Chessboard_11](https://github.com/user-attachments/assets/8d4d396f-b7f8-46cd-ad7e-1e4a7d26ca84)








