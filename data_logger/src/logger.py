#!/usr/bin/env python
import rospy
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
# experiences should have state, action, reward, state + 1
state_count = 0
current_state = [0] * 9
last_state = [0] * 9
reward = 0
reward_list = []
action = ""
action_space = ["l","r","s"]
replay_memory = []
pretrain_memory = []

# algorithm parameters
max_steps = 100
# learning_rate = 0.1 part of network?
total_reward = 0
ep_reward = 0
batch_size = 10
terminate_state = [0] * 9

#exploration parameters
explore_rate = 1
max_explore = 1
min_explore = .01
explore_decay = 0.01

is_training = True


#controller stuff
buttons = [0] * 11
move_cmd = Twist()

# TF stuff
class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


myAgent =  agent(lr = .01, s_size = 9, a_size = 3, h_size = 32) #is 9 correct
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

gradBuffer = sess.run(tf.trainable_variables())
for ix,grad in enumerate(gradBuffer):
    gradBuffer[ix] = grad * 0

def discount_rewards(r):
    gamma = 0.99 #discount future rewards - not part of network
    new_reward = 0
    discount_rewards = np.zeros_like(r)
    for i in reversed(xrange(0,r.size)):
        new_reward = new_reward * gamma + r[i]
        discount_rewards[i] = new_reward
    return discount_rewards

def pretrain_model():
    global action_space, pretrain_memory
    # randomize list of states/actions
    f = open("/home/rrc/path following testing area/training_data.txt", "r")
    preprocessed = [x.strip() for x in f]
    processed = []
    for i in range(0,len(preprocessed),2):
        state = list(map(int,list(str(preprocessed[i]))))
        processed.append((state,preprocessed[i+1]))
    random.shuffle(processed)
    f.close
    # loop through each state/acton pair
    for pretrain_state, pretrain_action in processed:
        a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[pretrain_state]})
        aInt = np.random.choice(a_dist[0],p=a_dist[0])
        aInt = np.argmax(a_dist == aInt)
        select_action = action_space[aInt]

        if select_action == pretrain_action:
            pretrain_reward = 100
        else:
            pretrain_reward = -100

        print(select_action)

        pretrain_memory.append([pretrain_state,aInt,pretrain_reward])

        # pass batch from replay memory to the networks
        pretrain_memory = np.array(pretrain_memory)
        pretrain_memory[:,2] = discount_rewards(pretrain_memory[:,2])
        feed_dict={myAgent.reward_holder:pretrain_memory[:,2],
                        myAgent.action_holder:pretrain_memory[:,1],myAgent.state_in:np.vstack(pretrain_memory[:,0])}
        grads = sess.run(myAgent.gradients, feed_dict=feed_dict)

        for idx,grad in enumerate(grads):
            gradBuffer[idx] += grad
        pretrain_memory = pretrain_memory.tolist()

pretrain_model()

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
        r = -10
    elif state[0] == 1 and state[2] == 1 and state[3] == 1 and state[5] == 1:
        r = 5
    elif not any(state):
        r = -100
    else:
        r = -1
    return r

def training(data):
    global state_count, current_state, last_state, total_reward
    global action, action_space, max_steps, learning_rate, replay_memory
    global batch_size, terminate_state, is_training, reward
    global explore_rate, max_explore, min_explore, explore_decay

    current_state = list(data.data)

    rospy.loginfo(current_state)

    if state_count >= max_steps or current_state == terminate_state:
        # how to pause and increment episode?
        # have a state selection variable
        explore_rate -= explore_decay
        rospy.loginfo("terminate this episode")
        is_training = False

    exploreChoice = random.uniform(0,1)
    aInt = 0
    if True:#exploreChoice > explore_rate: # we should exploit
        a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[current_state]})
        aInt = np.random.choice(a_dist[0],p=a_dist[0])
        aInt = np.argmax(a_dist == aInt)
        rospy.loginfo("state choice: " + str(aInt))
        action = action_space[aInt]
    else:
        # explore
        aInt = random.randint(0,2)
        action = action_space[aInt]

    # we then send the associated action to the turtlebot
    take_action(action)
    rospy.loginfo("action: " + action)

    if state_count > 0:
        reward = reward_function(current_state)
        total_reward += reward
        # should be last action, not current action
        if state_count<=100:
            replay_memory.append([last_state,aInt,reward,current_state])
        else:
            replay_memory[(state_count-1)%100] = [last_state,aInt,reward,current_state]
        # pass batch from replay memory to the networks
        replay_memory = np.array(replay_memory)
        replay_memory[:,2] = discount_rewards(replay_memory[:,2])
        feed_dict={myAgent.reward_holder:replay_memory[:,2],
                        myAgent.action_holder:replay_memory[:,1],myAgent.state_in:np.vstack(replay_memory[:,0])}
        grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
        for idx,grad in enumerate(grads):
            gradBuffer[idx] += grad
        replay_memory = replay_memory.tolist()

    state_count += 1
    last_state = current_state
    rospy.loginfo(state_count)

def standby(data):
    global explore_rate, explore_decay,min_explore
    global is_training, ep_reward, replay_memory
    rospy.loginfo("Awaiting restart")
    move_cmd.linear.x = 0
    move_cmd.angular.z = 0
    #restart on button press
    if buttons[2] == 1: #press 'x' to start new episode
        is_training = True
        ep_reward = 0
        replay_memory = []
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
    cmd_vel = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
    rate = rospy.Rate(3)
    rospy.Subscriber("joy", Joy, joy_callback) # sub to controller
    rospy.Subscriber("camera_state", UInt16MultiArray, callback)
    while not rospy.is_shutdown():
        rate.sleep()
        cmd_vel.publish(move_cmd)

if __name__ == '__main__':
    listener()
