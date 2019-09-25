#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


# In[13]:


def next_state(st, a):
    done = False
    if a == 0: #UP
        st[0] = st[0] - 1;
        if st[0] < 0:
            st[0] = 0;
    
    elif a == 1: #LEFT
        st[1] -= 1;
        if st[1] < 0:
            st[1] = 0;
    
    elif a == 2: #DOWN
        st[0] += 1;
        if st[0] > 3:
            st[0] = 3;
         
    else: #Right
        st[1] += 1;
        if st[1] > 3:
            st[1] = 3;
    
    #Chekcing for terminal state
    re = -1;
    if st[0] == 3 and st[1] == 3: #reached goal
        done = True
        
    elif st[1] > 0 and st[0] == 3: #fell of the cliff
        #m
        st[1] = 0;
        st[0] = 3;
        re = -100
        
    return st, re, done    


# In[27]:


#Q learning
goaly = 3
RR_QL = []
for runs in range(50):
    print("run: " , runs)
    episodes = 100
    Q = np.random.rand(4,4,4)
    #Terminal States
    Q[3,0,:] = 0
    Q[3,goaly,:] = 0
    Q[3,:,:] = 0

    #print("Q")
    #print(Q)
    
    alpha = 0.6
    epsilon = 0.1;
    gamma = 1;

    R = []
    for ep in range(episodes):
        #Start State
        s = [3, 0] 
        r = 0;
        Done = False
        r_sum = 0;
        while not Done:
            #print(s)
            
            if np.random.rand(1) < epsilon: #random action
                a = np.random.randint(0,4)

            else:
                a = np.argmax(Q[s[0],s[1],:])

            s_, r, Done = next_state(s, a)

            Q[s[0],s[1],a] = Q[s[0],s[1],a] + alpha*(r + gamma*np.max(Q[s_[0],s_[1],:]) - Q[s[0],s[1],a])
            #print(Q[s[0],s[1],a])
            
            
            alpha += 0.01
            if alpha > 1:
                alpha = 1;
                
            s = s_
            r_sum += r;
            
            if Done:
                if r_sum < -100:
                    r_sum = -100
                R.append(r_sum)
                
    
        #R = np.where(R < -100, -100, R)
        
    RR_QL.append(R)       
    
final_Q_learn = np.mean(RR_QL , axis = 0)


# In[24]:


print(Q[0][0][:])
np.argmax(Q[0][0][:])


# In[ ]:





# In[23]:


#SARSA
goaly = 3;
RR_SA = []
for runs in range(100):
    print("run: " , runs)
    episodes = 300
    Q = np.random.rand(4,4,4)
    #Terminal States
    Q[3,0,:] = 0 #START
    Q[3,goaly,:] = 0 #GOAL
    Q[3,:,:] = 0 #CLIFF

    alpha = 0.6
    epsilon = 0.1;
    gamma = 1
    
    R = []
    for ep in range(episodes):
        #Start State
        s = [3, 0] 
        r = 0;
        Done = False
        r_sum = 0;
        
        if np.random.rand(1) < epsilon: #random action
            a = np.random.randint(4)
        else:
            a = np.argmax(Q[s[0],s[1],:])
        
        while not Done:

            s_, r, Done = next_state(s, a)

            #finding the next action:
            if np.random.rand(1) < epsilon: #random action
                a_ = np.random.randint(4)
            else:
                a_ = np.argmax(Q[s_[0],s_[1],:])
                
                
            Q[s[0],s[1],a] = Q[s[0],s[1],a] + alpha*(r + gamma*Q[s_[0],s_[1],a_] - Q[s[0],s[1],a])

            s = s_
            a = a_
            r_sum += r;

            if Done:
                if r_sum < -100:
                    r_sum = -100;                
                
                R.append(r_sum)
        
        #R = np.where(np.array(R) < -100, -100, R)
    RR_SA.append(R)           
final_SARSA = np.mean(RR_SA , axis = 0)


# In[ ]:


final_SARSA


# In[ ]:


C = ('red' , 'blue')


X = np.arange(episodes)
lines = []
line_label = []
for i in range(2):
    if i == 0: #Q learnign
        lines.append(plt.plot(X, final_Q_learn, color=C[i]))
        line_label.append("Q learning")
    else:
        lines.append(plt.plot(X, final_SARSA, color=C[i]))
        line_label.append("SARSA")        
    
plt.legend(line_label)  
plt.savefig("Q7_rewards.png")
plt.show()


# In[ ]:




