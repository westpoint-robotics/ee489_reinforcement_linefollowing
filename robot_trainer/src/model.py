#!/usr/bin/env python
import rospy
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras import optimizers, regularizers,initializers
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from keras.constraints import max_norm
# experiences should have state, action, reward, state + 1
aInt = 0
state_count = 0
total_count = 0
replay_size = 100
state_count_array = []
state_count_in_prime = 0
state_count_in_prime_array = []
current_state = [0] * 9
last_state = [0] * 9
reward = 0
total_reward = 0
action = ''
action_space = ["l","s","r"]
replay_memory = []
pretrain_memory = []
ep_counter = 0
train_mode = True
trial = "Trial"

rewardOutput = open("/home/rrc/RL_Pathfollowing_Data/New_Expirement/6.67_percent/"+trial+"reward.txt","w+")
testOutput = open("/home/rrc/RL_Pathfollowing_Data/New_Expirement/6.67_percent/"+trial+"test.txt","w+")
# algorithm parameters
batch_size = 10
gamma = 0.95 #discount rate
terminate_state = [0] * 9

#exploration parameters
min_explore = 0.00
explore_decay = 0.0667
max_explore = 1 + explore_decay
explore_rate = max_explore

is_executing = True


#controller stuff
buttons = [0] * 11
move_cmd = Twist()

hiddenLayerDimension = 16  ##hidden layer dimension
model = Sequential()
model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=9))  ##input layer
for i in range(1):
    model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=hiddenLayerDimension,kernel_constraint=max_norm(1)))##hidden layers

#model.add(Dense(units=hiddenLayerDimension, activation='relu', kernel_regularizer=regularizers.l2(3),input_dim=hiddenLayerDimension)) ##regularize last hiddne layer

model.add(Dense(units=3, activation='softmax', input_dim=hiddenLayerDimension))##output layer
adam=optimizers.Adam(lr=0.001  , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam)

def pretrain_model():
    f = open("/home/rrc/path following testing area/training_data.txt", "r")
    preprocessed = [x.strip() for x in f]
    pretrain_states = np.empty((402,9))
    pretrain_actions = np.empty((402,3))
    for i in range(0,len(preprocessed),2):
        state = np.array(list(map(int,list(str(preprocessed[i])))))
        pretrain_states[int(i/2)] = state
        if preprocessed[i+1] == 'l':
            action = np.array([1,0,0])
        elif preprocessed[i+1] == 'r':
            action = np.array([0,0,1])
        else:
            action = np.array([0,1,0])
        pretrain_actions[int(i/2)] = action
    f.close
    history = model.fit(pretrain_states, pretrain_actions, validation_split=0.1,epochs=100,batch_size =20, verbose = 2)  ##train and validation is split over 90 % subset


#pretrain_model()
model._make_predict_function()
global graph
graph = tf.get_default_graph()

def take_action(a):
    if a == 's':
        move_cmd.linear.x = .3
        move_cmd.angular.z = 0
    elif a == 'r':
        move_cmd.linear.x = .3
        move_cmd.angular.z = -1.60
    elif a == 'l':
        move_cmd.linear.x = .3
        move_cmd.angular.z = 1.60
    else:
        move_cmd.linear.x = 0
        move_cmd.angular.z = 0


def joy_callback(data):
    global buttons
    buttons = data.buttons

def reward_function(state,action):
    r = 1 # weaker difference seems to help
    if state[7] == 1:
        r = 0
    elif not any(state):
        r = -1
    return r

def train_model(memory,batch_size):
    global model, gamma
    if batch_size > (np.size(memory,0)):
        return
    minibatch = random.sample(memory,batch_size)
    for state, action, reward, n_state in minibatch:
        with graph.as_default():
            target = reward
            #print(target, state, n_state)
            if reward != -1:
                target += gamma * np.amax(model.predict(np.array([n_state]))[0])
            target_f = model.predict(np.array([state]))
            target_f[0][action] = target
            #print(target_f[0])
            model.fit(np.array([state]),np.array(target_f),epochs=1, verbose=0)


def training(data):
    global state_count, current_state, last_state, aInt, state_count_in_prime
    global action, action_space, replay_memory, model, buttons, total_reward
    global batch_size, terminate_state, is_executing, reward, total_count
    global explore_rate, max_explore, min_explore, explore_decay, state_count_in_prime

    current_state = list(data.data)

    kill = False
    # press 'a' to terminate early
    if current_state == terminate_state or buttons[0] == 1:
        # how to pause and increment episode?
        # have a state selection variable
        rospy.loginfo("terminate this episode")
        is_executing = False
        kill = True

    exploreChoice = random.uniform(0,1)
    lastAInt = aInt
    aInt = 0
    if exploreChoice > explore_rate or train_mode == False: # we should exploit
        with graph.as_default():
            result = model.predict(np.array([current_state]))[0]
            aInt = np.argmax(result)
            print(result)
            print("I picked: " + action_space[aInt])
    else:
        # explore
        aInt = random.randint(0,2)

    action = action_space[aInt]
    # we then send the associated action to the turtlebot
    take_action(action)

    replay_memory = list(replay_memory)
    if state_count > 0:
        if not kill: reward = reward_function(current_state,lastAInt)
        else: reward = -1
        print(last_state,action_space[lastAInt],reward,current_state)
        # should be last action, not current action
        if train_mode == True:
            if total_count < replay_size:
                replay_memory.append([last_state,lastAInt,reward,current_state])
            else: replay_memory[total_count%replay_size] = [last_state,lastAInt,reward,current_state]
            # pass batch from replay memory to the networks

            replay_memory = np.array(replay_memory)
            train_model(replay_memory,batch_size)
            print("training")
            total_count += 1
            total_reward += reward
            rewardOutput.write("%d\n" % (total_reward))
        if reward == 1:
            state_count_in_prime += 1

    state_count += 1
    last_state = current_state
    if state_count == 65:
        is_executing = False

def standby(data):
    global explore_rate, explore_decay, min_explore, train_mode
    global is_executing, replay_memory, state_count, ep_counter, state_count_in_prime
    global state_count_array, state_count_in_prime_array
    move_cmd.linear.x = 0
    move_cmd.angular.z = 0
    #restart on button press
    if buttons[2] == 1: # press 'x' to resume training
        if train_mode == False:
            train_mode = True
            state_count_array.append(state_count)
            state_count_in_prime_array.append(state_count_in_prime)
            testOutput.write("Overall list of step counts: \n")
            for count in state_count_array:
                testOutput.write("%d\n" % (count))
            state_count_array[:]=[]
            testOutput.write("Overall list of good state step counts: \n")
            for count in state_count_in_prime_array:
                testOutput.write("%d\n" % (count))
            state_count_in_prime_array[:]=[]

        is_executing = True
        ep_counter += 1
        print("That was episode number: ",ep_counter)
        #state_count_array.append(state_count)
        #state_count_in_prime_array.append(state_count_in_prime)
        state_count = 0
        state_count_in_prime = 0
        if explore_rate-explore_decay >= min_explore:
            explore_rate -= explore_decay
            rospy.loginfo(explore_rate)
    elif buttons[3] == 1: # press 'y' to enter testing
    # it currently simply closes out the episode
        if train_mode == True:
            train_mode = False
        else:
            state_count_array.append(state_count)
            state_count_in_prime_array.append(state_count_in_prime)
        is_executing = True
        ep_counter += 1
        print("That was episode number: ",ep_counter)
        state_count = 0
        state_count_in_prime = 0

    elif buttons[1] == 1: # press 'b' to exit completely
        print("done")
        state_count_array.append(state_count)
        state_count_in_prime_array.append(state_count_in_prime)
        testOutput.write("Overall list of step counts: \n")
        for count in state_count_array:
            testOutput.write("%d\n" % (count))
        state_count_array[:]=[]
        testOutput.write("Overall list of good state step counts: \n")
        for count in state_count_in_prime_array:
            testOutput.write("%d\n" % (count))
        state_count_in_prime_array[:]=[]
        rewardOutput.close()
        testOutput.close()

def callback(data):
    if is_executing:
        training(data)
    else:
        standby(data)

def listener():
    global move_cmd
    rospy.init_node('listener',anonymous=True)
    cmd_vel = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
    rate = rospy.Rate(40)
    rospy.Subscriber("joy", Joy, joy_callback) # sub to controller
    rospy.Subscriber("camera_state", UInt16MultiArray, callback)
    while not rospy.is_shutdown():
        rate.sleep()
        cmd_vel.publish(move_cmd)

if __name__ == '__main__':
    listener()
