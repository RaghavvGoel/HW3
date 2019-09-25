#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


# In[3]:


d = float(6)
true_v = np.zeros(5)

for i in range(5):
    true_v[i] = i/d;
    


# In[4]:


#A = 1; B = 2; C = 3; D = 4; E = 5;
gamma = 1;
alpha = 0.1
v_final = []
episodes_range = [0, 1, 10, 100, 1000]
for episodes in episodes_range[:]:
    print("episode size: " , episodes)
    V = []
    v = 0.5*np.ones(7)
    v[0] = 0; v[6] = 0;
    
    for ep in range(episodes):
        print("ep= " , ep)
        
        #env reset
        s = 3; 
        Done = False;
        r = 0;
        while not Done:
            
            #take action, change state as MRP
            a = np.random.randint(2)
            if a == 0: #LEFT
                s_ = s - 1;
            else: #RIGHT
                s_ = s + 1;
            
            if s_ == 6:
                r = 1;
            else:
                r = 0;
            
            v[s] += alpha*(r + gamma*v[s_] - v[s])
            s = s_
            
            if s == 0 or s == 6:
                Done = True;
                V.append(v)
                #print(V)
                
    V = np.mean(V , axis=0)
    v_final.append(np.array(V)) 
    #print("v_final: " , v_final)
    print(" ")        
            


# In[5]:


v_final[0] = 0.5*np.ones(7)
v_final


# In[6]:


v_plot = np.zeros([np.size(episodes_range), 5])
for i in range(np.size(episodes_range)):
    for j in range(1,6):
        v_plot[i][j-1] = v_final[i][j]

v_plot[3]


# In[9]:


C = ('black' , 'red' , 'green' , 'blue' , 'purple')


X = np.arange(5)
lines = []
line_label = []
for i in range(np.size(episodes_range)):
    lines.append(plt.plot(X, v_plot[i], color=C[i]))
    line_label.append(str(episodes_range[i]))
    
plt.legend(line_label)    
plt.xlabel("STATES")
plt.ylabel("Expected Values")
plt.savefig("Q5_Estimated_Value.png")
plt.show()


# In[10]:


episodes = 100;
#true_v
alphas_TD = [0.15, 0.1, 0.05]
TD_e_a = []
for alpha in alphas_TD[:]:
    TD_e = []
    print("alpha: " , alpha)
    for run in range(101):
        #print(" ")
        
        #for episodes in range(1,101):
        E = [];
        v = 0.5*np.ones(7)
        v[0] = 0; v[6] = 0;
        
        for ep in range(100):
            #print("ep= " , ep)

            #env reset
            s = 3; 
            Done = False;
            r = 0;
            while not Done:

                #take action, change state as MRP
                a = np.random.randint(2)
                if a == 0: #LEFT
                    s_ = s - 1;
                else: #RIGHT
                    s_ = s + 1;

                if s_ == 6:
                    r = 1;
                else:
                    r = 0;

                v[s] += alpha*(r + gamma*v[s_] - v[s])
                s = s_

                if s == 0 or s == 6:
                    Done = True;

            e = np.power(np.average((v[1:6] - true_v)**2), 0.5)
            E.append(e);

        TD_e.append(E)
    #TD_run.append(TD_e)
    TD_e_a.append(np.mean(TD_e, axis = 0))


# In[11]:


np.shape(TD_e)
tmp = np.mean(TD_e, axis = 0)
print(np.shape(tmp))
v


# In[12]:


#MC

episodes = 100;
#true_v
alphas_MC = [0.01, 0.02, 0.03, 0.04]
MC_e_a = []
for alpha in alphas_MC[:]:
    print(" ")
    MC_e = []
    print("alpha: " , alpha)
    for run in range(100):
        #MC_e = []
        #for episodes in range(1,101):
        E = [];

        v = 0.5*np.ones(7)
        v[0] = 0; v[6] = 0;
        
        for ep in range(100):
            print("ep= " , ep)

            #env reset
            s = 3; 
            states = []
            Done = False;
            r = 0;
            G = 0;
            while not Done:
                #print("state: " , s)
                #take action, change state as MRP
                a = np.random.randint(2)
                #print("action: " a)
                if a == 0: #LEFT
                    s_ = s - 1;
                else: #RIGHT
                    s_ = s + 1;

                if s_ == 6:
                    r = 1;
                else:
                    r = 0;

                states.append(s)
                s = s_


                if s == 0 or s == 6:
                    Done = True;
                    if s == 0:
                        G = 0;
                    else:
                        G = 1;

                    for i in range(np.size(states)-1, -1 , -1):
                        j = states[i]
                        v[j] = v[j] + alpha*(G - v[j])

            e = np.power(np.average((v[1:6] - true_v)**2), 0.5)
            E.append(e);

        MC_e.append(E)

    MC_e_a.append(np.mean(MC_e , axis = 0))


# In[63]:


t = np.power(0.5*np.ones(5)**2 - true_v**2, 0.5)
np.average(t)
#np.size(MC_e_a[0])
v


# In[13]:


#PLOTTING RMS

C = ('black' , 'blue' , 'purple' ,  'red' , 'magenta' , 'green' , 'yellow')

# alphas_TD = [0.15, 0.1, 0.05]
# alphas_MC = [0.01, 0.02, 0.03, 0.04]

X = np.arange(100)
lines = []
line_label = []
for i in range(7):
    if i < 3:
        lines.append(plt.plot(X, TD_e_a[i], color=C[i]))
        line_label.append("TD_a=" + str(alphas_TD[i]))
    else:
        lines.append(plt.plot(X, MC_e_a[i-3], color=C[i]))
        line_label.append("MC_a=" + str(alphas_MC[i-3]))    
        
plt.legend(line_label)    
plt.xlabel("episodes")
plt.ylabel("RMS error")
plt.savefig("Q5_rms.png")
plt.show()


# In[69]:





# In[ ]:




