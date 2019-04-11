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
# experiences should have state, action, reward, state + 1
aInt = 0
state_count = 0
total_count = 0
current_state = [0] * 9
last_state = [0] * 9
reward = 0
reward_list = []
action = ''
action_space = ["l","r","s"]
replay_memory = []
pretrain_memory = []

# algorithm parameters
max_steps = 300
total_reward = 0
batch_size = 10
gamma = 0.95 #discount rate
terminate_state = [0] * 9

#exploration parameters
max_explore = .9
explore_rate = max_explore
min_explore = .01
explore_decay = 0.01

is_training = True


#controller stuff
buttons = [0] * 11
move_cmd = Twist()

hiddenLayerDimension = 16  ##hidden layer dimension
model = Sequential()
model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=9))  ##input layer
for i in range(1):
    model.add(Dense(units=hiddenLayerDimension, activation='relu', input_dim=hiddenLayerDimension))##hidden layers

#model.add(Dense(units=hiddenLayerDimension, activation='relu', kernel_regularizer=regularizers.l2(3),input_dim=hiddenLayerDimension)) ##regularize last hiddne layer

model.add(Dense(units=3, activation='softmax', input_dim=hiddenLayerDimension))##output layer
adam=optimizers.Adam(lr=0.001  , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam)

# def discount_rewards(r):
#     gamma = 0.99 #discount past rewards - not part of network
#     new_reward = 0
#     discount_rewards = np.zeros_like(r)
#     for i in reversed(xrange(0,r.size)):
#         new_reward = new_reward * gamma + r[i]
#         discount_rewards[i] = new_reward
#     return discount_rewards

def pretrain_model():
    f = open("/home/rrc/path following testing area/training_data.txt", "r")
    preprocessed = [x.strip() for x in f]
    pretrain_states = np.empty((402,9))
    pretrain_actions = np.empty((402,3))
    for i in range(0,len(preprocessed),2):
        state = np.array(list(map(int,list(str(preprocessed[i])))))
        pretrain_states[int(i/2)] = state
        if preprocessed[i+1] == 'l':
            action = np.array([5,0,0])
        elif preprocessed[i+1] == 'r':
            action = np.array([0,5,0])
        else:
            action = np.array([0,0,5])
        pretrain_actions[int(i/2)] = action
    f.close
    history = model.fit(pretrain_states, pretrain_actions, validation_split=0.1,epochs=100,batch_size =20, verbose = 2)  ##train and validation is split over 90 % subset


pretrain_model()
model._make_predict_function()
global graph
graph = tf.get_default_graph()

def take_action(a):
    if a == 's':
        move_cmd.linear.x = .15
        move_cmd.angular.z = 0
    elif a == 'r':
        move_cmd.linear.x = .15
        move_cmd.angular.z = -.80
    elif a == 'l':
        move_cmd.linear.x = .15
        move_cmd.angular.z = .80
    else:
        move_cmd.linear.x = 0
        move_cmd.angular.z = 0


def joy_callback(data):
    global buttons
    buttons = data.buttons

def reward_function(state):
    r = 0
    if state[7] == 1:
        r = -1
    elif not any(state):
        r = -100
    else:
        r = 5
    return r

# why do we need double brackets?
def train_model(memory,batch_size):
    global model, is_training, gamma
    if batch_size > np.size(2*memory,0):
        return
    minibatch = random.sample(memory,batch_size)
    for state, action, reward, n_state in minibatch:
        with graph.as_default():
            target = reward
            #print(target, state, n_state)
            if reward != 0:
                target += gamma * np.amax(model.predict(np.array([n_state]))[0])
            target_f = model.predict(np.array([state]))
            target_f[0][action] = target
            #print(target_f[0])
            model.fit(np.array([state]),np.array(target_f),epochs=1, verbose=0)

def training(data):
    global state_count, current_state, last_state, total_reward, aInt
    global action, action_space, max_steps, replay_memory, model
    global batch_size, terminate_state, is_training, reward, total_count
    global explore_rate, max_explore, min_explore, explore_decay

    current_state = list(data.data)

    rospy.loginfo(current_state)

    if state_count >= max_steps or current_state == terminate_state:
        # how to pause and increment episode?
        # have a state selection variable
        rospy.loginfo("terminate this episode")
        is_training = False

    exploreChoice = random.uniform(0,1)
    lastAInt = aInt
    aInt = 0
    if 1: #exploreChoice > explore_rate: # we should exploit
        with graph.as_default():
            result = model.predict(np.array([current_state]))[0]
            aInt = np.argmax(result)
            expectedValue = np.amax(result)
            rospy.loginfo(result)
    else:
        # explore
        aInt = random.randint(0,2)

    action = action_space[aInt]

    # we then send the associated action to the turtlebot
    take_action(action)
    rospy.loginfo("action: " + action)

    replay_memory = list(replay_memory)
    if state_count > 0:
        reward = reward_function(current_state)
        print("reward: ", reward)
        total_reward += reward
        # should be last action, not current action
        if total_count<300:
            replay_memory.append([last_state,lastAInt,reward,current_state])
        else:
            replay_memory[(total_count)%300] = [last_state,lastAInt,reward,current_state]
        # pass batch from replay memory to the networks
        replay_memory = np.array(replay_memory)
        train_model(replay_memory,10)
        #replay_memory[:,2] = discount_rewards(replay_memory[:,2])

    state_count += 1
    total_count += 1
    last_state = current_state

def standby(data):
    global explore_rate, explore_decay,min_explore
    global is_training, replay_memory, state_count
    move_cmd.linear.x = 0
    move_cmd.angular.z = 0
    #restart on button press
    if buttons[2] == 1: #press 'x' to start new episode
        is_training = True
        state_count = 0
        if explore_rate > min_explore:
            explore_rate -= explore_decay

def callback(data):
    if is_training:
        training(data)
    else:
        standby(data)

def listener():
    global move_cmd
    rospy.init_node('listener',anonymous=True)
    cmd_vel = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
    rate = rospy.Rate(3)
    rospy.Subscriber("joy", Joy, joy_callback) # sub to controller
    rospy.Subscriber("camera_state", UInt16MultiArray, callback)
    while not rospy.is_shutdown():
        rate.sleep()
        cmd_vel.publish(move_cmd)

if __name__ == '__main__':
    listener()
