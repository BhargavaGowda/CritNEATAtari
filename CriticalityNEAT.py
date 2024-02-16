import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import neat
import pickle
import visualize
import matplotlib.pyplot as plt
import numpy as np


env  =  gym.make("ALE/Breakout-ram-v5",full_action_space=True)
env = FlattenObservation(env)
observation, info = env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    "configAtari.txt")

popSize = 30
gens = 100
runsteps = 300

def main():

    #setup
    genomeConfig = config.genome_config
    pop = []
    metric1 = []
    fitnessList = []
    bestFitness = -10000
    bestGenome = None
    fitCurve = np.zeros(gens)
    popMaxCurve = np.zeros(gens)
    numRolloutsPerEval = 1

    for i in range(popSize):
        newGenome = neat.DefaultGenome(i)
        newGenome.configure_new(genomeConfig)
        newGenome.fitness = 0
        pop.append(newGenome)

    #Warmup
        
    for i in range(popSize):
        genome = pop[i]

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        fitness = 0
        worstFitness = 10000
        runNum = 1
        step = 0
        observation, info = env.reset()
        obsSize = observation.size
        
        # m1 = np.array([])
        m1 = np.zeros(obsSize)
        m2 = np.zeros(obsSize)
        m3 = np.zeros(obsSize)


        #up,down,left,right,fire
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
            if terminated or truncated:
                    reward = 0
            step+=1
            fitness += reward
            m1 += 0.1*(observation-m1)
            m2 += 0.01*(observation-m2)
            m3 += 0.001*(observation-m3)

            # if step%20 == 0:
            #     m1 = np.concatenate((m1,observation,[fitness]))


            if step>runsteps:
                runNum+=1
                
                if fitness< worstFitness:
                    worstFitness = fitness
                fitness = 0
                step = 0
                
                observation, info = env.reset()
            
            if runNum>numRolloutsPerEval:
                break


        # print(m1)
        # print(m2)
        # print(m3)
        metric1.append(np.concatenate((m1,m2,m3)))
        fitnessList.append(fitness)


    # for i in range(min(10,popSize)):
    #     print(pop[i].fitness,metric1[i])


    for gen in range(gens):

        print("running gen:", gen)
        for g in range(popSize):

            parent1 = pop[g]
            parent2 = None
            parent2index = 0
            bestCriticality = 0
            for g2 in range(popSize):
                if g!=g2:
                    genomicDist = parent1.distance(pop[g2],genomeConfig)
                    behaviourDist = np.linalg.norm(np.array(metric1[g])-np.array(metric1[g2]))
                    criticality = behaviourDist/genomicDist
                    if criticality>bestCriticality:
                        parent2 = pop[g2]
                        parent2index = g2
                        bestCriticality = criticality
            
            #print("p1:",g,"p2:",parent2index)
            if not parent2:
                raise RuntimeError("no parent2")
            



            testGenome = neat.DefaultGenome(g)
            testGenome.configure_crossover(parent1,parent2,genomeConfig)
            testGenome.mutate(genomeConfig)
            testGenome.fitness = 0

            pop[g] = testGenome

            net = neat.nn.FeedForwardNetwork.create(testGenome, config)
            fitness = 0
            worstFitness = 10000
            step = 0
            runNum = 1
            observation, info = env.reset()
            # m1 = np.array([])
            m1 = np.zeros(obsSize)
            m2 = np.zeros(obsSize)
            m3 = np.zeros(obsSize)

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
                step+=1
                if terminated or truncated:
                    reward = 0
                fitness += reward
                m1 += 0.1*(observation-m1)
                m2 += 0.01*(observation-m2)
                m3 += 0.001*(observation-m3)
                # if step%20 == 0:
                #     m1 = np.concatenate((m1,observation,[fitness]))
                
                if step>runsteps:
                    runNum+=1
                    
                    if fitness< worstFitness:
                        worstFitness = fitness
                    fitness = 0
                    step = 0
                    observation, info = env.reset()

                if runNum>numRolloutsPerEval:
                    break     

            metric1[g] = np.concatenate((m1,m2,m3))
            fitnessList[g] = worstFitness
        

            if worstFitness>bestFitness:
                bestFitness = worstFitness
                bestGenome = testGenome
                bestGenomeM1 = m1

        #reporter
        print("best in current pop:", np.max(fitnessList) ,"best overall:", bestFitness)
        popMaxCurve[gen] = np.max(fitnessList)
        fitCurve[gen] = bestFitness
        
        if gen%100 == 0 and gen>0:
            with open("gen"+str(gen)+"_CheckpointCrit.pkl", "wb") as f:
                pickle.dump(pop, f)




    print("BestFitness:", bestFitness)
    visualize.draw_net(config, bestGenome, True)

    
    

    with open("bestGenomeCrit.pkl", "wb") as f:
        pickle.dump(bestGenome, f)


    with open("lastPopCrit.pkl", "wb") as f:
        pickle.dump(pop, f)

    plt.plot(fitCurve)
    plt.plot(popMaxCurve)
    plt.show()


    

main()