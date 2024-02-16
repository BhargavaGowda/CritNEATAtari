import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle
import numpy as np
import visualize

env  =  gym.make("ALE/Breakout-ram-v5",full_action_space=True,render_mode = "human")
env = FlattenObservation(env)
observation, info = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "configAtari.txt")


with open("bestGenomeCrit.pkl", "rb") as f:
    genome = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(genome, config)
print(genome.size())
visualize.draw_net(config, genome, True)


numRuns = 20

fitnessList=np.zeros(numRuns)
fitness = 0
runNum = 0
while True:

    output = net.activate(observation)
    action = np.zeros(env.action_space.shape)
    
    if output[0]>0.5:
        #up
        if output[2]>0.5:
            #upleft
            if output[4]>0.5:
                action = 15
            else:
                action = 7
        elif output[3]>0.5:
            #upright
            if output[4]>0.5:
                action = 14
            else:
                action = 6
        else:
            #just up
            if output[4]>0.5:
                action = 10
            else:
                action = 2
    elif output[1]>0.5:
        #down
        if output[2]>0.5:
            #downleft
            if output[4]>0.5:
                action = 17
            else:
                action = 9
        elif output[3]>0.5:
            #downright
            if output[4]>0.5:
                action = 16
            else:
                action = 8
        else:
            #just down
            if output[4]>0.5:
                action = 13
            else:
                action = 5
    else:
        #neutral updown
        if output[2]>0.5:
            #left
            if output[4]>0.5:
                action = 12
            else:
                action = 4
        elif output[3]>0.5:
            #right
            if output[4]>0.5:
                action = 11
            else:
                action = 3
        else:
            #neutral
            if output[4]>0.5:
                action = 1
            else:
                action = 0

    observation, reward, terminated, truncated, info = env.step(action)
    fitness+=reward
    if terminated or truncated:
        observation, info = env.reset()
        print("Run:",runNum,"Fitness:",fitness)
        fitnessList[runNum]=fitness
        runNum+=1
        fitness = 0
    
    if runNum>=numRuns:
        break

print("maxFitness:",np.max(fitnessList))

env.close()