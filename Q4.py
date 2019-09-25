#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


def card_value(card, sum_):
    #print("card is: " , card)
    if card > 9:
        return 10
    elif card == 1:
        if sum_ < 11:
            return 11; #
        else:
            return 1;
    else:
        return card;


# In[3]:


def player_take_action(p_sum, usable_ace):
    
    p_bust = False;
    p_done = False;
    
    if p_sum < 20:
        #HIT
        val_ = card_value(np.random.randint(13) + 1, p_sum);
        if val_ == 11:
            usable_ace = 1;
        #print("val_ = " , val_)    
        p_sum += val_
        #print("player_sum new: " , p_sum)
        
        if p_sum > 21 and usable_ace == 1:
            p_sum = p_sum - 10;
            usable_ace = 0;
            
        elif p_sum > 21 and usable_ace == 0: #STICK
            p_bust = True;
            p_done = True;
            
    else:
        p_done = True;
        
    
    return p_sum, p_done, p_bust, usable_ace


# In[4]:


def dealer_take_action(d_sum):
    
    d_bust = False;
    d_done = False;
    if d_sum < 17:
        #HIT
        d_sum += card_value(np.random.randint(13) + 1, d_sum)
        if d_sum > 21:
            d_done = True
            d_bust = True
    else:
        d_done = True
        
    return d_sum, d_done, d_bust    


# In[5]:


episodes = int(50e+4)
gamma = 1;


# In[7]:


v = np.zeros([10, 10, 2])
N = np.zeros([10, 10, 2])
#returns = [[] for ]

for ep in range(1,episodes+1):
    print("ep: " , ep)
    #reset env
    Done = False;
    G = 0; #do we even need this, yes coz of subsequent states but all are same, no discount
    r = 0;
    #generating random walk:
    seq_states = []
    #geneate initial state:
    dealer_card = card_value(np.random.randint(13) + 1, 0)
    if dealer_card == 11:
        dealer_card = 1;
    player_sum = np.random.randint(10) + 12
    usable_ace = np.random.randint(2) #where 0 means no usable ace, and 1 means usable ace
    
    seq_states.append([dealer_card-1, player_sum-12, usable_ace])
    #print("state for ep:" , ep , )
    #print(dealer_card , " " , player_sum , " " , usable_ace)
    #print(player_sum)
    #print(usable_ace)
    #print(" ")
    #let's take a walk now, as discount=1, this means that all the states will get the same reward: 1,-1 or 0
    #depending on whether the player wins or loses or draws
    while not Done:
        #PLayers plays:
        player_done = False
        while not player_done:
            player_new_sum, player_done, player_bust, usable_ace = player_take_action(player_sum, usable_ace)
            #print("player_new_sum: " , player_new_sum)
            
            if not player_done:
                seq_states.append([dealer_card-1, player_new_sum-12, usable_ace])
                player_new_sum = player_sum;
                
                
            #print("player_done: " , player_done)
            #print("player_bust: " , player_bust)
        #Dealer plays:
        
        #print("Dealer Turn")
        if not player_bust:
            dealer_sum = dealer_card
            if dealer_card == 1:
                dealer_sum = 11;
            dealer_done = False
            while not dealer_done:    
                dealer_new_sum, dealer_done, dealer_bust = dealer_take_action(dealer_sum)
                dealer_sum = dealer_new_sum
                #print("dealer sum : " , dealer_sum)
       
        Done = True;
        if player_bust:
            r = -1;
        elif dealer_bust:
            r = 1;
        else:
            if player_sum > dealer_sum:
                r = 1;
            elif player_sum == dealer_sum:    
                r = 0;
            else:
                r = -1;
    
    #print(seq_states)
    #print("reward: " , r , " shape: " , np.shape(seq_states))
    #print(" ")
    #traversing over the sequence generated:
    for i in range(np.shape(seq_states)[0]-1,-1,-1):
        s = seq_states[i]
        #print("s: " , s , " i: " , i)
        if (np.shape(seq_states)[0] > 1 and s not in seq_states[:i]) or np.shape(seq_states)[0] == 1:
            #print("cur val: " , v[s[0],s[1],s[2]])
            N[s[0],s[1],s[2]] += 1
            v[s[0],s[1],s[2]] += (1/float(N[s[0],s[1],s[2]]))*(r - v[s[0],s[1],s[2]])
            #v[s[0],s[1],s[2]] 
            
    #print(v[:,:,0])
    #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    



# In[ ]:





# In[6]:


#v[:,:,0]


# In[8]:


#refer here: https://stackoverflow.com/questions/11766536/matplotlib-3d-surface-from-a-rectangular-array-of-heights

X = np.arange(0, 10)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
#print(X)
#print(Y)
Z = np.array(v[:,:,0])
#print(Z)
#print(np.shape(v) , " ", np.shape(v[:,:,0]))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("withoutAce_500kTrials.png")
plt.show()


# In[9]:


X = np.arange(0, 10)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
#print(X)
#print(Y)
Z = np.array(v[:,:,1])
#print(Z)
#print(np.shape(v) , " ", np.shape(v[:,:,0]))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig("withAce_500kTrials.png")

plt.show()


# In[6]:


def player_take_action_ES(p_sum, usable_ace):
    
    p_bust = False;
    p_done = False;
    
    a = np.random.randint(2)
    
    if a == 0: #Stick
        p_done = True;
    
    else: #HIT
    #if p_sum < 20:
        #HIT
        val_ = card_value(np.random.randint(13) + 1, p_sum);
        if val_ == 11:
            usable_ace = 1;
        
        #p_sum updated --------------------!
        p_sum += val_ 
        #print("val_ = " , val_)
        #print("player_sum new: " , p_sum)
        
        #Checking for bust, and preventing it
        if p_sum > 21 and usable_ace == 1:
            p_sum = p_sum - 10;
            usable_ace = 0;
            
        elif p_sum > 21 and usable_ace == 0: #STICK
            p_bust = True;
            p_done = True;
        
    
    return p_sum, p_done, p_bust, usable_ace, a


# In[7]:


p = 0.5*np.ones([10,10,2]) #equi probable
q = np.zeros([10,10,2, 2]) #first 3 for states
N = np.zeros([10,10,2,2])


# In[9]:



for ep in range(1,episodes+1):
    print("ep: " , ep)
    #reset env
    Done = False;
    G = 0; #do we even need this, yes coz of subsequent states but all are same, no discount
    r = 0;
    #generating random walk:
    seq_sa = []
    #geneate initial state:
    dealer_card = card_value(np.random.randint(13) + 1, 0)
    if dealer_card == 11:
        dealer_card = 1;
    player_sum = np.random.randint(10) + 12
    usable_ace = np.random.randint(2) #where 0 means no usable ace, and 1 means usable ace
    
    #Selecting action for player randomly: hit or stick : 1 and 0 resp.
    a = np.random.randint(2)
    
    #seq_s_a.append([dealer_card-1, player_sum-12, usable_ace, a])
    #print("state for ep:" , ep , )
    #print(dealer_card , " " , player_sum , " " , usable_ace, " " , a)
    #print(player_sum)
    #print(usable_ace)
    #print(" ")
    #let's take a walk now, as discount=1, this means that all the states will get the same reward: 1,-1 or 0
    #depending on whether the player wins or loses or draws
    while not Done:
        #player checks whether action is sticking or hitting
        player_done = False
        player_bust = False
        if a == 0:
            player_done = True;
        
        
        #PLayers plays if not done:    
        while not player_done:
            player_new_sum, player_done, player_bust, usable_ace, a = player_take_action_ES(player_sum, usable_ace)
            #print("player_new_sum: " , player_new_sum)
            
            #if not player_done:
            seq_sa.append([dealer_card-1, player_sum-12, usable_ace, a]) 
            #appending prev state with action, which leads us to next state
            player_new_sum = player_sum;
            
            #if player goes bust, then player done, then we'll come out of the while
                
            #print("player_done: " , player_done)
            #print("player_bust: " , player_bust)
        #Dealer plays:
        
        #print("Dealer Turn")
        if not player_bust:
            dealer_sum = dealer_card
            if dealer_card == 1:
                dealer_sum = 11;
            dealer_done = False
            while not dealer_done:    
                dealer_new_sum, dealer_done, dealer_bust = dealer_take_action(dealer_sum)
                dealer_sum = dealer_new_sum
                #print("dealer sum : " , dealer_sum)
       
        Done = True;
        if player_bust:
            r = -1;
        elif dealer_bust:
            r = 1;
        else:
            if player_sum > dealer_sum:
                r = 1;
            elif player_sum == dealer_sum:    
                r = 0;
            else:
                r = -1;
    
    print(seq_sa)
    #print("reward: " , r , " shape: " , np.shape(seq_sa))
    #print(" ")
    #traversing over the sequence generated:
    for i in range(np.shape(seq_sa)[0]-1,-1,-1):
        sa = seq_sa[i]
        #print("s: " , s , " i: " , i)
        if (sa not in seq_sa[:i] and np.shape(seq_sa)[0] > 1) or np.shape(seq_sa)[0] == 1:
            N[sa[0],sa[1],sa[2],sa[3]] += 1;
            q[sa[0],sa[1],sa[2],sa[3]] += 1/float(N[sa[0],sa[1],sa[2],sa[3]])*(r - q[sa[0],sa[1],sa[2],sa[3]]);
            
            #q[sa[0],sa[1],sa[2],sa[3]] /= float(ep)
            #we should also average over the polciy selected, so that it becomes stochastic
            p[sa[0],sa[1],sa[2]] = np.argmax(q[sa[0],sa[1],sa[2],:])

            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            
#p[sa[0],sa[1],sa[2]] /= float(ep)
            
    #print(q[:,:,0])
    
    



# In[237]:


p[:,:,0]


# In[10]:


#PLOTTING POLICY!!
X = np.arange(0, 10)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
#print(X)
#print(Y)
Z = np.array(p[:,:,0])
#print(Z)
#print(np.shape(v) , " ", np.shape(v[:,:,0]))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("withoutAce_policy.png")
plt.show()


# In[11]:


#PLOTTING POLICY!!
X = np.arange(0, 10)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
#print(X)
#print(Y)
Z = np.array(p[:,:,1])
#print(Z)
#print(np.shape(v) , " ", np.shape(v[:,:,0]))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("withAce_policy.png")
plt.show()


# In[24]:


#FOR V_star, traverse over all the Q values and take the max!!
v_star = np.zeros([10,10,2])

for i in range(10):
    for j in range(10):
        v_star[i][j][0] = np.max(q[i,j,0,:])
        v_star[i][j][1] = np.max(q[i,j,1,:])
    
# print(np.shape(v_star[:,:,0]))
# print(v_star[:,:,0])
# print(q[:,:,0,:])


# In[29]:


#plot this
X = np.arange(0, 10)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
#print(X)
#print(Y)
Z = np.array(v_star[:,:,0])
#print(Z)
#print(np.shape(v) , " ", np.shape(v[:,:,0]))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlim("dealer card")
# plt.ylim("sum")
plt.savefig("withoutAce_v_star.png")
plt.show()


# In[30]:


#plot this
X = np.arange(0, 10)
Y = np.arange(0, 10)
X, Y = np.meshgrid(X, Y)
#print(X)
#print(Y)
Z = np.array(v_star[:,:,1])
#print(Z)
#print(np.shape(v) , " ", np.shape(v[:,:,0]))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlim("dealer card")
# plt.ylim("sum")
plt.savefig("withAce_v_star.png")
plt.show()


# In[34]:


def player_take_action_Without_ES(p_sum, usable_ace):
    
    p_bust = False;
    p_done = False;
    
    a = np.random.randint(2)
    
    if a == 0: #Stick
        p_done = True;
    
    else: #HIT
    #if p_sum < 20:
        #HIT
        val_ = card_value(np.random.randint(13) + 1, p_sum);
        if val_ == 11:
            usable_ace = 1;
        
        #p_sum updated --------------------!
        p_sum += val_ 
        #print("val_ = " , val_)
        #print("player_sum new: " , p_sum)
        
        #Checking for bust, and preventing it
        if p_sum > 21 and usable_ace == 1:
            p_sum = p_sum - 10;
            usable_ace = 0;
            
        elif p_sum > 21 and usable_ace == 0: #STICK
            p_bust = True;
            p_done = True;
        
    
    return p_sum, p_done, p_bust, usable_ace


# In[35]:


#OFF POLICY

b = np.ones([10,10,2]) #behaviour policy
t = np.ones([10,10,2]) #target policy
t[:,9:,:] = 0; #sticking if 20 or 21


# In[36]:


v_ori = 0; 
v_weighted = 0;
target_value = -0.27726;

final_w = []
final_o = []

episode_list = [1, 10, 100, 1000, 10000, 100000]
for episodes in episode_list:
    J = []
    G = []
    for ep in range(1,episodes+1):
        print("ep: " , ep)
        #reset env
        Done = False;
        #G = 0; #do we even need this, yes coz of subsequent states but all are same, no discount
        r = 0;
        #generating random walk:
        seq_sa = []
        #geneate initial state:
        dealer_card = 2;
        player_sum = 13;

        usable_ace = 0;#np.random.randint(2) #where 0 means no usable ace, and 1 means usable ace

        
        
        #Selecting action for player randomly: hit or stick : 1 and 0 resp.
        a = np.random.randint(2)

        #seq_s_a.append([dealer_card-1, player_sum-12, usable_ace, a])
        print("state for ep:" , ep , )
        print(dealer_card , " " , player_sum , " " , usable_ace)
        #print(player_sum)
        #print(usable_ace)
        print(" ")
        #let's take a walk now, as discount=1, this means that all the states will get the same reward: 1,-1 or 0
        #depending on whether the player wins or loses or draws
        step = 0;
        while not Done:
            #player checks whether action is sticking or hitting
            player_done = False
            if a == 0:
                player_done = True;

            #PLayers plays if not done:    
            while not player_done:
                player_new_sum, player_done, player_bust, usable_ace = player_take_action_Without_ES(player_sum, usable_ace)
                #print("player_new_sum: " , player_new_sum)

                #if not player_done:
                seq_sa.append([dealer_card-1, player_sum-12, usable_ace]) 
                #appending prev state with action, which leads us to next state
                player_new_sum = player_sum;

                #if player goes bust, then player done, then we'll come out of the while

                #print("player_done: " , player_done)
                #print("player_bust: " , player_bust)
            #Dealer plays:

            #print("Dealer Turn")
            if not player_bust:
                dealer_sum = dealer_card
                if dealer_card == 1:
                    dealer_sum = 11;
                dealer_done = False
                while not dealer_done:    
                    dealer_new_sum, dealer_done, dealer_bust = dealer_take_action(dealer_sum)
                    dealer_sum = dealer_new_sum
                    #print("dealer sum : " , dealer_sum)

            Done = True;
            step += 1;
            if player_bust:
                r = -1;
            elif dealer_bust:
                r = 1;
            else:
                if player_sum > dealer_sum:
                    r = 1;
                elif player_sum == dealer_sum:    
                    r = 0;
                else:
                    r = -1;

        G.append(r)
        if ep > 1:
            #J.append(step + J[ep-1])
            J.append(step)
        else:
            J.append(step)

        print(seq_sa)
        print("reward: " , r , " shape: " , np.shape(seq_sa))
        print(" ")

    #Ordinary:
    tmp_n = 0; tmp_d_w = 0; tmp_d_o = 0;
    for i,j in enumerate(J):
            tmp_n += (1/0.5)**(j)*G[i]
            tmp_d_w += (1/0.5)**(j)

    tmp_d_o = np.size(J)

    v_ori = tmp_n/float(tmp_d_o)
    v_weighted = tmp_n/float(tmp_d_w)

    rms_err_o = (v_ori - target_value)**2    
    rms_err_w = (v_weighted - target_value)**2    

    final_o.append(rms_err_o)
    final_w.append(rms_err_w)
    


# In[37]:


final_o


# In[258]:


final_w


# In[39]:


C = ('red' , 'blue')


X = np.arange(np.size(final_w))
lines = []
line_label = []
for i in range(2):
    if i == 0: #Q learnign
        lines.append(plt.plot(X, final_o, color=C[i]))
        line_label.append("oridanary")
    else:
        lines.append(plt.plot(X, final_w, color=C[i]))
        line_label.append("weighted")        
    
plt.legend(line_label)  
plt.savefig("Q4_weighted.png")
plt.show()


# In[ ]:




