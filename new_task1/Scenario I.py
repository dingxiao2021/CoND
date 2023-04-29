import networkx as nx
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import argparse
import time
import copy
import pickle
import os
import datetime
import pandas as pd
from torch_geometric.nn import DenseGINConv,DenseGCNConv
from torch.nn import Sequential, Linear, ReLU, LSTM, BatchNorm1d 
use_cuda=True
device = torch.device("cuda" if use_cuda else "cpu")
# read network and dynamics
def readnet_dynamic(net_name, N):   
    G0 = nx.Graph()
    G0.add_nodes_from(range(0,N))
    
    name_path = r'data/'+net_name+'/'+net_name
    f = open(name_path+'.txt')               
    lines = f.readlines()          
    for line in lines:             
        listline1 = line.strip('\n').split(' ')[0] 
        listline2 = line.strip('\n').split(' ')[1]
        G0.add_edges_from([(int(listline1), int(listline2))])
    f.close()
    
    # read dynamics  
    x1 = np.loadtxt(name_path+'_threshold_x.txt', dtype=np.float32, delimiter=' ')  # T*N
    y1 = np.loadtxt(name_path+'_threshold_y.txt', dtype=np.float32, delimiter=' ')  
    x2 = np.loadtxt(name_path+'_voter_x.txt', dtype=np.float32, delimiter=' ')  # T*N
    y2 = np.loadtxt(name_path+'_voter_y.txt', dtype=np.float32, delimiter=' ') 
    x3 = np.loadtxt(name_path+'_ising_x.txt', dtype=np.float32, delimiter=' ')  # T*N
    y3 = np.loadtxt(name_path+'_ising_y.txt', dtype=np.float32, delimiter=' ') 
    x4 = np.loadtxt(name_path+'_SIS_x.txt', dtype=np.float32, delimiter=' ')  # T*N
    y4 = np.loadtxt(name_path+'_SIS_y.txt', dtype=np.float32, delimiter=' ') 
    x5 = np.loadtxt(name_path+'_SIR_x.txt', dtype=np.float32, delimiter=' ')  # T*N
    y5 = np.loadtxt(name_path+'_SIR_y.txt', dtype=np.float32, delimiter=' ') 
    
    
    return G0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, name_path

#read data
#net_name='BA_N200_m2' N=200
#net_name='ER_N200_p002' N=200
#net_name='WS_N200_k4_p05' N=200
#net_name='dolphins' N=62
#net_name='parsed_word' N=112
#net_name='ca-netscience' N=379
#net_name='email' N=1133

net_name='BA_N200_m2'
N=200
G0, X_th, Y_th, X_voter, Y_voter, X_ising, Y_ising, X_sis, Y_sis, X_sir, Y_sir, net_path = readnet_dynamic(net_name, N)

def RemoveRandomEdges(G,n_edges):
    for i in range(n_edges):
        rdm_edges_id = np.random.choice(range(G.number_of_edges()))
        edges = list(G.edges())
        e = edges[rdm_edges_id]
        G.remove_edge(*e)
    return G
      
    
class GCNNet1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNNet1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc1 = nn.Sequential(
            torch.nn.Linear(in_channels, 32),
            torch.nn.ReLU())
        self.conv = DenseGCNConv(32, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = nn.Linear(32, out_channels)
    def forward(self, x, adj):
        x = self.fc1(x)
        x = self.relu(self.conv(x, adj))
        x = self.fc2(x)
        
        return x
class GumbelGraphNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GumbelGraphNetwork, self).__init__()
        self.edge1 = torch.nn.Linear(2*input_size, hidden_size)
        self.edge2edge = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(input_size+hidden_size, output_size * input_size)
        
    def forward(self, x, adj):
        # adj[batch,node,node]
        # x[batch,node,dim]
        out = x
        # innode [batch,node,node,dim]
        innode = out.unsqueeze(1).repeat(1, adj.size()[1], 1, 1)
        # outnode [batch,node,node,dim]
        outnode = innode.transpose(1, 2)
        # node2edge:[batch,node,node,2*dim->hidden]
        node2edge = F.relu(self.edge1(torch.cat((innode,outnode), 3)))
        # edge2edge:[batch,node,node,hidden]
        edge2edge = F.relu(self.edge2edge(node2edge))
        # adjs:[batch,node,node,1]
        adjs = adj.view(adj.size()[0], adj.size()[1], adj.size()[2], 1)
        # adjs:[batch,node,node,hidden]
        adjs = adjs.repeat(1, 1, 1, edge2edge.size()[3])
        # edges:[batch,node,node,hidden]
        edges = adjs * edge2edge
        # out:[batch,node,hid]
        out = torch.sum(edges, 2)
        # out:[batch,node,hid]
        out = F.relu(self.node2node(out))
        out = F.relu(self.node2node2(out))
        # out:[batch,node,dim+hid]
        out = torch.cat((x, out), dim = -1)
        # out:[batch,node,dim]
        out = self.output(out)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

def gumbel_sample(shape, eps=1e-20):
    u = torch.rand(shape)
    gumbel = - np.log(- np.log(u + eps) + eps)
    gumbel = gumbel.to(device)
    return gumbel
def gumbel_softmax_sample(logits, temperature): 
    y = logits + gumbel_sample(logits.size())
    return torch.nn.functional.softmax( y / temperature, dim = 1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        #k = logits.size()[-1]
        y_hard = torch.max(y.data, 1)[1]
        y = y_hard
    return y
class Gumbel_Generator(nn.Module):
    def __init__(self, sz = 10, temp = 10, temp_drop_frac = 0.9999):
        super(Gumbel_Generator, self).__init__()
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
        #gen_matrix 为邻接矩阵的概率
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac
    def drop_temperature(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac
    def sample(self, hard=False):
        # 采样——得到一个临近矩阵
        self.logp = self.gen_matrix.view(-1, 2)
        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
            out = out.to(device)
        out_matrix = out[:,0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        #out_matrix=(out_matrix+out_matrix.T)/2
        return out_matrix
    def get_temperature(self):
        return self.temperature
    def get_cross_entropy(self, obj_matrix):
        # 计算与目标矩阵的距离
        logps = F.softmax(self.gen_matrix, 2)
        logps = torch.log(logps[:,:,0] + 1e-10) * obj_matrix + torch.log(logps[:,:,1] + 1e-10) * (1 - obj_matrix)
        result = - torch.sum(logps)
        result = result.cpu() if use_cuda else result
        return result.data.numpy()
    def get_entropy(self):
        logps = F.softmax(self.gen_matrix, 2)
        result = torch.mean(torch.sum(logps * torch.log(logps + 1e-10), 1))
        result = result.cpu() if use_cuda else result
        return(- result.data.numpy())
    def randomization(self, fraction):
        # 将gen_matrix重新随机初始化，fraction为重置比特的比例
        sz = self.gen_matrix.size()[0]
        numbers = int(fraction * sz * sz)
        original = self.gen_matrix.cpu().data.numpy()
        
        for i in range(numbers):
            ii = np.random.choice(range(sz), (2, 1))
            z = torch.rand(2).cuda() if use_cuda else torch.rand(2)
            self.gen_matrix.data[ii[0], ii[1], :] = z

class Generator_states(nn.Module):
    def __init__(self,dat_num,del_num):
        super(Generator_states, self).__init__()
        self.embeddings = nn.Embedding(dat_num, del_num)
    def forward(self, idx):
        pos_probs = torch.sigmoid(self.embeddings(idx).unsqueeze(2))
        return pos_probs
def train_batch_dyn1(a,w1,optimizer_dyn,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,r2,r1,t):
    optimizer_dyn.zero_grad()
    unstates=states_learner(t)
    data_train[:,r1,:]=unstates[:,r1,:]
    adj=adj*a+w1
    adj = adj.unsqueeze(0)
    adj = adj.repeat(data_train.size()[0],1,1)
    #get result caculated by neural network
    output = dyn_learner(data_train,adj)
    output = output.permute(0,2,1)
    data_target = data_target.long()
    loss = loss_fn(output[:,:,r2],data_target[:,r2])
    loss.backward() 
    optimizer_dyn.step()
    return loss
def train_batch_states1(a,w1,opt_states,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,loss_fn1,r2,r1,t):
    opt_states.zero_grad()
    unstates=states_learner(t)
    data_train[:,r1,:]=unstates[:,r1,:]
    out_matrix=adj*a+w1
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    # get result caculated by neural network
    output = dyn_learner(data_train,out_matrix)
    output = output.permute(0,2,1)
    # caculate the difference
    data_target = data_target.long()
    loss = loss_fn(output[:,:,r2],data_target[:,r2])#+loss_fn1(unstates[:,r2,:],data_train[:,r2,:])
    loss.backward() 
    opt_states.step()
    return loss
def train_batch_generator1(a,w1,states_learner,optimizer_network,gumbel_generator,dyn_learner,data_train,data_target,loss_fn,r2,r1,t):
    optimizer_network.zero_grad()
    unstates=states_learner(t)
    data_train[:,r1,:]=unstates[:,r1,:]
    p = gumbel_generator.sample()
    out_matrix=w1+p*a
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    gumbel_generator.drop_temperature()
    # get result caculated by neural network
    output = dyn_learner(data_train,out_matrix)
    output = output.permute(0,2,1)
    # caculate the difference
    data_target = data_target.long()
    loss = loss_fn(output[:,:,r2],data_target[:,r2])
    loss=loss#+(1/(len(w1)*len(w1)))*torch.norm(p,p=1)
    loss.backward()
    optimizer_network.step()
    return loss
def train_batch_dyn2(w1,optimizer_dyn,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,r2,r1,t):
    optimizer_dyn.zero_grad()
    unstates=states_learner(t)
    data_train[:,r1,:]=unstates[:,r1,:]
    adj=w1+adj
    adj = adj.unsqueeze(0)
    adj = adj.repeat(data_train.size()[0],1,1)
    #get result caculated by neural network
    output = dyn_learner(data_train,adj)
    output = output.permute(0,2,1)
    data_target = data_target.long()
    loss = loss_fn (output[:,:,r2],data_target[:,r2])
    loss.backward() 
    optimizer_dyn.step()
    return loss
def train_batch_states2(w1,opt_states,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,loss_fn1,r2,r1,t):
    opt_states.zero_grad()
    unstates=states_learner(t)
    data_train[:,r1,:]=unstates[:,r1,:]
    out_matrix=w1+adj
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    # get result caculated by neural network
    output = dyn_learner(data_train,out_matrix)
    output = output.permute(0,2,1)
    # caculate the difference
    data_target = data_target.long()
    loss = loss_fn(output[:,:,r2],data_target[:,r2])+loss_fn1(unstates[:,r2,:],data_train[:,r2,:])
    loss.backward() 
    opt_states.step()
    return loss
def train_batch_generator2(w1,states_learner,optimizer_network,gumbel_generator,dyn_learner,data_train,data_target,loss_fn,r2,r1,t):
    optimizer_network.zero_grad()
    unstates=states_learner(t)
    data_train[:,r1,:]=unstates[:,r1,:]
    p = gumbel_generator.sample()
    out_matrix=p+w1
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    gumbel_generator.drop_temperature()
    # get result caculated by neural network
    output = dyn_learner(data_train,out_matrix)
    output = output.permute(0,2,1)
    # caculate the difference
    data_target = data_target.long()
    loss = loss_fn(output[:,:,r2],data_target[:,r2])
    loss=loss+(1/(len(w1)*len(w1)))*torch.norm(p,p=1)
    loss.backward()
    optimizer_network.step()
    return loss
def train_batch_dyn3(w1,optimizer_dyn,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,r2,r1,t):
    optimizer_dyn.zero_grad()
    unstates=states_learner(t)
    data_train[:,r1,:]=unstates[:,r1,:]
    adj=w1
    adj = adj.unsqueeze(0)
    adj = adj.repeat(data_train.size()[0],1,1)
    #get result caculated by neural network
    output = dyn_learner(data_train,adj)
    output = output.permute(0,2,1)
    data_target = data_target.long()
    loss = loss_fn (output[:,:,r2],data_target[:,r2])
    loss.backward() 
    optimizer_dyn.step()
    return loss
def train_batch_states3(w1,opt_states,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,loss_fn1,r2,r1,t):
    opt_states.zero_grad()
    unstates=states_learner(t)
    data_train[:,r1,:]=unstates[:,r1,:]
    out_matrix=w1
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data_train.size()[0], 1, 1)
    # get result caculated by neural network
    output = dyn_learner(data_train,out_matrix)
    output = output.permute(0,2,1)
    # caculate the difference
    data_target = data_target.long()
    loss = loss_fn(output[:,:,r2],data_target[:,r2])+loss_fn1(unstates[:,r2,:],data_train[:,r2,:])
    loss.backward() 
    opt_states.step()
    return loss

batch=100
batch_num=9
W0 = nx.to_numpy_array(G0)

num_nodes = len(W0)


#dyn_names = ['SIR']
#Xs = [X_sir]
#Ys = [Y_sir]

dyn_names = ['threshold','voter','ising','SIS','SIR']
Xs = [X_th, X_voter, X_ising, X_sis, X_sir]
Ys = [Y_th, Y_voter, Y_ising, Y_sis, Y_sir]
net_path = r'result\\'+net_name
del_net1=[0.1,0.2]
for del_net in del_net1:
    for i in range(len(dyn_names)):
        dyn_name = dyn_names[i]
        X = Xs[i]
        Y = Ys[i]
        X1=X.T
        X2=X1.copy()
        Y1=Y.T
        Y2=Y1.copy()
        result=[]
        result1=[]
        out_channels = 2
        if dyn_name == 'SIR':
           out_channels = 3
           X2[X2==1]=0.5
           X2[X2==2]=1
        for k in range(10):   
            #随机删边
            n=len(np.array(G0.edges))
            dddd=del_net
            s=n*dddd
            G1=RemoveRandomEdges(G0.copy(),int(s))#部分结构
            len(G1.edges)

            W0 = nx.to_numpy_array(G0)
            W1 = nx.to_numpy_array(G1)
            batch_num=9
            #删除几个点
            n=dddd*N
            s1=[]
            s2=[]
            r1=random.sample(range(G1.number_of_nodes()),int(n))
            for i in range(batch_num):
                #删除的点
                #保留下来的点
                r2=[i for i in range(G1.number_of_nodes())]
                for i in r1:
                    #print(i)
                    r2.remove(i)
                #len(r2)
                s1.append(r1)
                s2.append(r2)
            batch=100
            num_nodes = len(W1)
            use_cuda=True
            device = torch.device("cuda" if use_cuda else "cpu")
            states_learner=Generator_states(batch*(batch_num+1),num_nodes).to(device)
            opt_states = optim.Adam(states_learner.parameters(), lr=0.1)
            t=torch.Tensor([i for i in range(batch_num*batch,(batch_num+1)*batch)]).long().to(device)
            dingxiao=states_learner(t)

            use_cuda=True
            device = torch.device("cuda" if use_cuda else "cpu")

            num_nodes = len(W1)
            in_channels = 1
            hidden_channels =32
            #out_channels = 32
            #动力学学习器
            dyn_learner = GumbelGraphNetwork(in_channels, hidden_channels, out_channels).to(device)
            #dyn_learner = dyn_learner.double()
            optimizer_dyn = optim.Adam(dyn_learner.parameters(),lr = 0.001)

            #网络学习器
            gumbel_generator = Gumbel_Generator(sz = num_nodes,temp = 10,temp_drop_frac = 0.9999).to(device)
            #gumbel_generator = gumbel_generator.double()
            optimizer_network = optim.Adam(gumbel_generator.parameters(),lr = 0.1)

            states_learner=Generator_states(batch*(batch_num+1),num_nodes).to(device)
            opt_states = optim.Adam(states_learner.parameters(), lr=0.1)


            #部分结构
            w1=torch.tensor(W1)
            w1=w1.float()
            w1=w1.to(device)

            a=W1.copy()
            a[a>0]=-1
            a=torch.tensor(a+1).to(device).float()

            #损失函数
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn1 = torch.nn.MSELoss()
            #总迭代次数
            Epoch_Num=10
            #动力学学习次数
            Dyn_Steps=20
            #网络学习次数
            Net_Steps=10
            Sta_Steps=30
            #批数



            loss = 0
            dyn_losses = []
            net_losses = []
            accu_record = []
            err_nets = []
            acc_nets = []
            # start training
            losses = []
            accuracies = []
            losses_in_gumbel = []
            losses_state=[]
            for epoch in range(Epoch_Num):
                print('epoch running:'+str(epoch)+' / '+str(Epoch_Num))
                print('use gumbel')
                adj = gumbel_generator.sample(hard=True)
                # 先训练dynamics

                # dyn_learner.train()
                print("\n***************Dyn Training******************")
                for i in range(Dyn_Steps):
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss = train_batch_dyn1(a,w1,optimizer_dyn,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses.append(record_loss)

                print("\n***************State Training******************")
                for i in range(Sta_Steps):
                    step_loss = 0
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss= train_batch_states1(a,w1,opt_states,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,loss_fn1,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses_state.append(record_loss)

                print("\n***************Gumbel Training******************")
                for i in range(Net_Steps):
                    step_loss = 0
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss= train_batch_generator1(a,w1,states_learner,optimizer_network,gumbel_generator,dyn_learner,data_train,data_target,loss_fn,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses_in_gumbel.append(record_loss)

            #利用部分结构+P计算动力学acc
            pp=gumbel_generator.sample()
            acc3=0
            z=0
            t=torch.Tensor([i for i in range(batch_num*batch,(batch_num+1)*batch)]).long().to(device)
            unstates=states_learner(t)
            X3=X2.copy()
            Y3=Y2.copy()
            data_train=torch.Tensor(X3[:,batch_num*batch:(batch_num+1)*batch].T)[:,:,None]
            data_target=torch.Tensor(Y3[:,batch_num*batch:(batch_num+1)*batch].T)
            data_train=data_train.to(device)
            data_target=data_target.to(device)
            #data_train[:,s1[0],:]=unstates[:,s1[0],:]

            print('###########################')
            for t in range(0,100):
                z=z+1
                x_real=data_train[t,:,:][None,:,:]
                y_real=data_target[t,:][None,:].T

                y_real=y_real.view(1, -1)[0].long()
                adj=w1+pp*a
                adj = adj.unsqueeze(0)
                adj = adj.repeat(1, 1, 1)
                x_real=x_real.to(device)        
                y_pred_1 = dyn_learner(x_real, adj)

                #print(y_pred.shape)
                y_pred_2=F.softmax(y_pred_1,2)

                p1=torch.argmax(y_pred_2, axis=2)
                t1=y_real

                t2=p1[:,:]

                for i in range(len(t1)):
                    if t2[:,i]==t1[i]:
                        acc3=acc3+1

            print("\n***************动力学预测ACC******************")
            print(acc3/(len(t1)*z))

            loss_state1=0
            loss_state11=0
            acc_state1=0
            for i in range(len(s1[0])):
                t=torch.Tensor([k for k in range(batch*batch_num,(batch*(batch_num+1)))]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,(batch*batch_num):(batch*(batch_num+1))]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state1=loss_state1+loss_fn2(dddd,ffff)
                loss_state11=loss_state11+loss_fn2(dddd,ffff)/(batch)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state1=acc_state1+1
            from scipy import sparse
            pp=pp.cpu()
            p=W1+pp.detach().numpy()
            perturbation =p

            is_directed = False
            if is_directed == False:
                perturbation = (perturbation+perturbation.T)/2
            else:
                perturbation = perturbation

            for i in range(0,perturbation.shape[0]):
                perturbation[i][i]=0

            perturbation[W1>0] = 0
            a=sparse.dok_matrix(perturbation)
            b = sorted(a.items(), key=lambda x: x[1], reverse=True)
            c=sparse.dok_matrix(W0-W1)
            d = sorted(c.items(), key=lambda x: x[1], reverse=True)

            A=[]
            B=[]
            #s：删除了多少的边
            #把边放入一个list
            for i in range(int(s)*2):
                A.append(b[i][0])
            for i in range(int(s)*2):
                B.append(d[i][0])
            #取交集
            print("\n***************网络推断ACC******************")
            print('acc:',len(set(A)&set(B))/(int(s)*2))
            import AUC
            auc2=AUC.Calculation_AUC(W1,W0-W1,W1+pp.detach().numpy(),len(W1))
            print("\n***************网络推断AUC******************")
            print('auc:',auc2)


            pp=gumbel_generator.sample()
            pp=pp.cpu()
            perturbation=pp.detach().numpy()
            perturbation = (perturbation+perturbation.T)/2
            for i in range(0,perturbation.shape[0]):
                perturbation[i][i]=0

            perturbation[W1>0] = 0
            from sklearn.metrics import roc_curve, auc
            fpr_gnn1, tpr_gnn1, _ = roc_curve((W0-W1).flatten().astype(np.int32),perturbation.flatten())
            auc1=auc(fpr_gnn1, tpr_gnn1)
            print(auc1)
            loss_state=0
            loss_state1=0
            acc_state=0
            for i in range(len(s1[0])):
                t=torch.Tensor([i for i in range(0,batch*batch_num)]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,0:batch*batch_num]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state=loss_state+loss_fn2(dddd,ffff)
                loss_state1=loss_state1+loss_fn2(dddd,ffff)/(batch*batch_num)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state=acc_state+1
            print(loss_state)
            print(loss_state1)
            print(acc_state/(batch*batch_num*len(s1[0])))
            result.append([acc3/(len(w1)*z),len(set(A)&set(B))/(int(s)*2),auc2,auc1,loss_state.item(),acc_state/(batch*batch_num*len(s1[0])),loss_state1.item(),acc_state1/(batch*len(s1[0]))])

            use_cuda=True
            device = torch.device("cuda" if use_cuda else "cpu")

            num_nodes = len(W1)
            in_channels = 1
            #out_channels = 2
            #动力学学习器
            dyn_learner = GCNNet1(in_channels, out_channels).to(device)
            #dyn_learner = dyn_learner.double()
            optimizer_dyn = optim.Adam(dyn_learner.parameters(),lr = 0.001)

            #网络学习器
            gumbel_generator = Gumbel_Generator(sz = num_nodes,temp = 10,temp_drop_frac = 0.9999).to(device)
            #gumbel_generator = gumbel_generator.double()
            optimizer_network = optim.Adam(gumbel_generator.parameters(),lr = 0.1)

            states_learner=Generator_states(batch*(batch_num+1),num_nodes).to(device)
            opt_states = optim.Adam(states_learner.parameters(), lr=0.1)


            #部分结构
            w1=torch.tensor(W1)
            w1=w1.float()
            w1=w1.to(device)

            #损失函数
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn1 = torch.nn.MSELoss()
            #总迭代次数
            Epoch_Num=10
            #动力学学习次数
            Dyn_Steps=20
            #网络学习次数
            Net_Steps=10
            Sta_Steps=30
            #批数



            loss = 0
            dyn_losses = []
            net_losses = []
            accu_record = []
            err_nets = []
            acc_nets = []
            # start training
            losses = []
            accuracies = []
            losses_in_gumbel = []
            losses_state=[]
            for epoch in range(Epoch_Num):
                print('epoch running:'+str(epoch)+' / '+str(Epoch_Num))
                print('use gumbel')
                adj = gumbel_generator.sample(hard=True)
                # 先训练dynamics

                # dyn_learner.train()
                print("\n***************Dyn Training******************")
                for i in range(Dyn_Steps):
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss = train_batch_dyn2(w1,optimizer_dyn,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses.append(record_loss)

                print("\n***************State Training******************")
                for i in range(Sta_Steps):
                    step_loss = 0
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss= train_batch_states2(w1,opt_states,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,loss_fn1,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses_state.append(record_loss)

                print("\n***************Gumbel Training******************")
                for i in range(Net_Steps):
                    step_loss = 0
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss= train_batch_generator2(w1,states_learner,optimizer_network,gumbel_generator,dyn_learner,data_train,data_target,loss_fn,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses_in_gumbel.append(record_loss)

            #利用部分结构+P计算动力学acc
            pp=gumbel_generator.sample()
            #adj1 = gumbel_generator.sample(hard=True)
            acc3=0
            z=0
            t=torch.Tensor([i for i in range(batch_num*batch,(batch_num+1)*batch)]).long().to(device)
            unstates=states_learner(t)
            X3=X2.copy()
            Y3=Y2.copy()
            data_train=torch.Tensor(X3[:,batch_num*batch:(batch_num+1)*batch].T)[:,:,None]
            data_target=torch.Tensor(Y3[:,batch_num*batch:(batch_num+1)*batch].T)
            data_train=data_train.to(device)
            data_target=data_target.to(device)
            #data_train[:,s1[0],:]=unstates[:,s1[0],:]

            print('###########################')
            for t in range(0,100):
                z=z+1
                x_real=data_train[t,:,:][None,:,:]
                y_real=data_target[t,:][None,:].T

                y_real=y_real.view(1, -1)[0].long()
                adj=w1+pp
                adj = adj.unsqueeze(0)
                adj = adj.repeat(1, 1, 1)
                x_real=x_real.to(device)        
                y_pred_1 = dyn_learner(x_real, adj)

                #print(y_pred.shape)
                y_pred_2=F.softmax(y_pred_1,2)

                p1=torch.argmax(y_pred_2, axis=2)
                t1=y_real

                t2=p1[:,:]

                for i in range(len(t1)):
                    if t2[:,i]==t1[i]:
                        acc3=acc3+1

            print("\n***************动力学预测ACC******************")
            print(acc3/(len(t1)*z))

            loss_state1=0
            loss_state11=0
            acc_state1=0
            for i in range(len(s1[0])):
                t=torch.Tensor([k for k in range(batch*batch_num,(batch*(batch_num+1)))]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,(batch*batch_num):(batch*(batch_num+1))]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state1=loss_state1+loss_fn2(dddd,ffff)
                loss_state11=loss_state11+loss_fn2(dddd,ffff)/(batch)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state1=acc_state1+1
            from scipy import sparse
            pp=pp.cpu()
            p=W1+pp.detach().numpy()
            
            perturbation =p

            is_directed = False
            if is_directed == False:
                perturbation = (perturbation+perturbation.T)/2
            else:
                perturbation = perturbation

            for i in range(0,perturbation.shape[0]):
                perturbation[i][i]=0
            
            perturbation[W1>0] = 0
            a=sparse.dok_matrix(perturbation)
            b = sorted(a.items(), key=lambda x: x[1], reverse=True)
            c=sparse.dok_matrix(W0-W1)
            d = sorted(c.items(), key=lambda x: x[1], reverse=True)

            A=[]
            B=[]
            #s：删除了多少的边
            #把边放入一个list
            for i in range(int(s)*2):
                A.append(b[i][0])
            for i in range(int(s)*2):
                B.append(d[i][0])
            #取交集
            print("\n***************网络推断ACC******************")
            print('acc:',len(set(A)&set(B))/(int(s)*2))
            import AUC
            auc2=AUC.Calculation_AUC(W1,W0-W1,W1+pp.detach().numpy(),len(W1))
            print("\n***************网络推断AUC******************")
            print('auc:',auc2)


            pp=gumbel_generator.sample()
            pp=pp.cpu()
            perturbation=pp.detach().numpy()
            perturbation = (perturbation+perturbation.T)/2
            for i in range(0,perturbation.shape[0]):
                perturbation[i][i]=0

            perturbation[W1>0] = 0
            from sklearn.metrics import roc_curve, auc
            fpr_gnn2, tpr_gnn2, _ = roc_curve((W0-W1).flatten().astype(np.int32),perturbation.flatten())
            auc1=auc(fpr_gnn2, tpr_gnn2)
            print(auc1)
            loss_state=0
            loss_state1=0
            acc_state=0
            for i in range(len(s1[0])):
                t=torch.Tensor([i for i in range(0,batch*batch_num)]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,0:batch*batch_num]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state=loss_state+loss_fn2(dddd,ffff)
                loss_state1=loss_state1+loss_fn2(dddd,ffff)/(batch*batch_num)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state=acc_state+1
            print(loss_state)
            print(loss_state1)
            print(acc_state/(batch*batch_num*len(s1[0])))
            result.append([acc3/(len(w1)*z),len(set(A)&set(B))/(int(s)*2),auc2,auc1,loss_state.item(),acc_state/(batch*batch_num*len(s1[0])),loss_state1.item(),acc_state1/(batch*len(s1[0]))])




            use_cuda=True
            device = torch.device("cuda" if use_cuda else "cpu")

            num_nodes = len(W1)
            in_channels = 1
            #out_channels = 2
            #动力学学习器
            dyn_learner = GCNNet1(in_channels, out_channels).to(device)
            #dyn_learner = dyn_learner.double()
            optimizer_dyn = optim.Adam(dyn_learner.parameters(),lr = 0.001)

            #网络学习器
            gumbel_generator = Gumbel_Generator(sz = num_nodes,temp = 10,temp_drop_frac = 0.9999).to(device)
            #gumbel_generator = gumbel_generator.double()
            optimizer_network = optim.Adam(gumbel_generator.parameters(),lr = 0.1)

            states_learner=Generator_states(batch*(batch_num+1),num_nodes).to(device)
            opt_states = optim.Adam(states_learner.parameters(), lr=0.1)


            #部分结构
            w1=torch.tensor(W1)
            w1=w1.float()
            w1=w1.to(device)

            #损失函数
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn1 = torch.nn.MSELoss()
            #总迭代次数
            Epoch_Num=10
            #动力学学习次数
            Dyn_Steps=20
            #网络学习次数
            Net_Steps=10
            Sta_Steps=30
            #批数



            loss = 0
            dyn_losses = []
            net_losses = []
            accu_record = []
            err_nets = []
            acc_nets = []
            # start training
            losses = []
            accuracies = []
            losses_in_gumbel = []
            losses_state=[]
            for epoch in range(Epoch_Num):
                print('epoch running:'+str(epoch)+' / '+str(Epoch_Num))
                print('use gumbel')
                adj = gumbel_generator.sample(hard=True)
                # 先训练dynamics

                # dyn_learner.train()
                print("\n***************Dyn Training******************")
                for i in range(Dyn_Steps):
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss = train_batch_dyn2(w1,optimizer_dyn,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses.append(record_loss)

                print("\n***************Gumbel Training******************")
                for i in range(Net_Steps):
                    step_loss = 0
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss= train_batch_generator2(w1,states_learner,optimizer_network,gumbel_generator,dyn_learner,data_train,data_target,loss_fn,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses_in_gumbel.append(record_loss)

            #利用部分结构+P计算动力学acc
            pp=gumbel_generator.sample()
            acc3=0
            z=0
            t=torch.Tensor([i for i in range(batch_num*batch,(batch_num+1)*batch)]).long().to(device)
            unstates=states_learner(t)
            X3=X2.copy()
            Y3=Y2.copy()
            data_train=torch.Tensor(X3[:,batch_num*batch:(batch_num+1)*batch].T)[:,:,None]
            data_target=torch.Tensor(Y3[:,batch_num*batch:(batch_num+1)*batch].T)
            data_train=data_train.to(device)
            data_target=data_target.to(device)
            #data_train[:,s1[0],:]=unstates[:,s1[0],:]+1

            print('###########################')
            for t in range(0,100):
                z=z+1
                x_real=data_train[t,:,:][None,:,:]
                y_real=data_target[t,:][None,:].T

                y_real=y_real.view(1, -1)[0].long()
                adj=w1+pp
                adj = adj.unsqueeze(0)
                adj = adj.repeat(1, 1, 1)
                x_real=x_real.to(device)        
                y_pred_1 = dyn_learner(x_real, adj)

                #print(y_pred.shape)
                y_pred_2=F.softmax(y_pred_1,2)

                p1=torch.argmax(y_pred_2, axis=2)
                t1=y_real

                t2=p1[:,:]

                for i in range(len(t1)):
                    if t2[:,i]==t1[i]:
                        acc3=acc3+1

            print("\n***************动力学预测ACC******************")
            print(acc3/(len(t1)*z))

            loss_state1=0
            loss_state11=0
            acc_state1=0
            for i in range(len(s1[0])):
                t=torch.Tensor([k for k in range(batch*batch_num,(batch*(batch_num+1)))]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,(batch*batch_num):(batch*(batch_num+1))]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state1=loss_state1+loss_fn2(dddd,ffff)
                loss_state11=loss_state11+loss_fn2(dddd,ffff)/(batch)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state1=acc_state1+1
            from scipy import sparse
            pp=pp.cpu()
            p=W1+pp.detach().numpy()
            perturbation =p

            is_directed = False
            if is_directed == False:
                perturbation = (perturbation+perturbation.T)/2
            else:
                perturbation = perturbation

            for i in range(0,perturbation.shape[0]):
                perturbation[i][i]=0

            perturbation[W1>0] = 0
            a=sparse.dok_matrix(perturbation)
            b = sorted(a.items(), key=lambda x: x[1], reverse=True)
            c=sparse.dok_matrix(W0-W1)
            d = sorted(c.items(), key=lambda x: x[1], reverse=True)

            A=[]
            B=[]
            #s：删除了多少的边
            #把边放入一个list
            for i in range(int(s)*2):
                A.append(b[i][0])
            for i in range(int(s)*2):
                B.append(d[i][0])
            #取交集
            print("\n***************网络推断ACC******************")
            print('acc:',len(set(A)&set(B))/(int(s)*2))
            import AUC
            auc2=AUC.Calculation_AUC(W1,W0-W1,W1+pp.detach().numpy(),len(W1))
            print("\n***************网络推断AUC******************")
            print('auc:',auc2)


            pp=gumbel_generator.sample()
            pp=pp.cpu()
            perturbation=pp.detach().numpy()
            perturbation = (perturbation+perturbation.T)/2
            for i in range(0,perturbation.shape[0]):
                perturbation[i][i]=0

            perturbation[W1>0] = 0
            from sklearn.metrics import roc_curve, auc
            fpr_gnn2, tpr_gnn2, _ = roc_curve((W0-W1).flatten().astype(np.int32),perturbation.flatten())
            auc1=auc(fpr_gnn2, tpr_gnn2)
            print(auc1)
            loss_state=0
            loss_state1=0
            acc_state=0
            for i in range(len(s1[0])):
                t=torch.Tensor([i for i in range(0,batch*batch_num)]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,0:batch*batch_num]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state=loss_state+loss_fn2(dddd,ffff)
                loss_state1=loss_state1+loss_fn2(dddd,ffff)/(batch*batch_num)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state=acc_state+1
            print(loss_state)
            print(loss_state1)
            print(acc_state/(batch*batch_num*len(s1[0])))
            result.append([acc3/(len(w1)*z),len(set(A)&set(B))/(int(s)*2),auc2,auc1,loss_state.item(),acc_state/(batch*batch_num*len(s1[0])),loss_state1.item(),acc_state1/(batch*len(s1[0]))])








            use_cuda=True
            device = torch.device("cuda" if use_cuda else "cpu")

            num_nodes = len(W1)
            in_channels = 1
            #out_channels = 2
            #动力学学习器
            dyn_learner = GCNNet1(in_channels, out_channels).to(device)
            #dyn_learner = dyn_learner.double()
            optimizer_dyn = optim.Adam(dyn_learner.parameters(),lr = 0.001)

            #网络学习器
            gumbel_generator = Gumbel_Generator(sz = num_nodes,temp = 10,temp_drop_frac = 0.9999).to(device)
            #gumbel_generator = gumbel_generator.double()
            optimizer_network = optim.Adam(gumbel_generator.parameters(),lr = 0.1)

            states_learner=Generator_states(batch*(batch_num+1),num_nodes).to(device)
            opt_states = optim.Adam(states_learner.parameters(), lr=0.1)


            #部分结构
            w1=torch.tensor(W1)
            w1=w1.float()
            w1=w1.to(device)

            #损失函数
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn1 = torch.nn.MSELoss()
            #总迭代次数
            Epoch_Num=10
            #动力学学习次数
            Dyn_Steps=20
            #网络学习次数
            Net_Steps=10
            Sta_Steps=30
            #批数



            loss = 0
            dyn_losses = []
            net_losses = []
            accu_record = []
            err_nets = []
            acc_nets = []
            # start training
            losses = []
            accuracies = []
            losses_in_gumbel = []
            losses_state=[]
            for epoch in range(Epoch_Num):
                print('epoch running:'+str(epoch)+' / '+str(Epoch_Num))
                print('use gumbel')
                adj = gumbel_generator.sample(hard=True)
                # 先训练dynamics

                # dyn_learner.train()
                print("\n***************Dyn Training******************")
                for i in range(Dyn_Steps):
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss = train_batch_dyn3(w1,optimizer_dyn,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses.append(record_loss)

                print("\n***************State Training******************")
                for i in range(Sta_Steps):
                    step_loss = 0
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss= train_batch_states3(w1,opt_states,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,loss_fn1,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses_state.append(record_loss)

                print("\n***************Gumbel Training******************")

            #利用部分结构+P计算动力学acc
            pp=gumbel_generator.sample()
            acc3=0
            z=0
            t=torch.Tensor([i for i in range(batch_num*batch,(batch_num+1)*batch)]).long().to(device)
            unstates=states_learner(t)
            X3=X2.copy()
            Y3=Y2.copy()
            data_train=torch.Tensor(X3[:,batch_num*batch:(batch_num+1)*batch].T)[:,:,None]
            data_target=torch.Tensor(Y3[:,batch_num*batch:(batch_num+1)*batch].T)
            data_train=data_train.to(device)
            data_target=data_target.to(device)
            #data_train[:,s1[0],:]=unstates[:,s1[0],:]

            print('###########################')
            for t in range(0,100):
                z=z+1
                x_real=data_train[t,:,:][None,:,:]
                y_real=data_target[t,:][None,:].T

                y_real=y_real.view(1, -1)[0].long()
                adj=w1
                adj = adj.unsqueeze(0)
                adj = adj.repeat(1, 1, 1)
                x_real=x_real.to(device)        
                y_pred_1 = dyn_learner(x_real, adj)

                #print(y_pred.shape)
                y_pred_2=F.softmax(y_pred_1,2)

                p1=torch.argmax(y_pred_2, axis=2)
                t1=y_real

                t2=p1[:,:]

                for i in range(len(t1)):
                    if t2[:,i]==t1[i]:
                        acc3=acc3+1

            print("\n***************动力学预测ACC******************")
            print(acc3/(len(t1)*z))

            loss_state1=0
            loss_state11=0
            acc_state1=0
            for i in range(len(s1[0])):
                t=torch.Tensor([k for k in range(batch*batch_num,(batch*(batch_num+1)))]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,(batch*batch_num):(batch*(batch_num+1))]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state1=loss_state1+loss_fn2(dddd,ffff)
                loss_state11=loss_state11+loss_fn2(dddd,ffff)/(batch)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state1=acc_state1+1

            loss_state=0
            loss_state1=0
            acc_state=0
            for i in range(len(s1[0])):
                t=torch.Tensor([i for i in range(0,batch*batch_num)]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,0:batch*batch_num]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state=loss_state+loss_fn2(dddd,ffff)
                loss_state1=loss_state1+loss_fn2(dddd,ffff)/(batch*batch_num)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state=acc_state+1
            print(loss_state)
            print(loss_state1)
            print(acc_state/(batch*batch_num*len(s1[0])))
            result1.append([acc3/(len(w1)*z),loss_state.item(),acc_state/(batch*batch_num*len(s1[0])),loss_state1.item(),acc_state1/(batch*len(s1[0]))])


            use_cuda=True
            device = torch.device("cuda" if use_cuda else "cpu")

            num_nodes = len(W1)
            in_channels = 1
            #out_channels = 2
            #动力学学习器
            dyn_learner = GCNNet1(in_channels, out_channels).to(device)
            #dyn_learner = dyn_learner.double()
            optimizer_dyn = optim.Adam(dyn_learner.parameters(),lr = 0.001)

            #网络学习器
            gumbel_generator = Gumbel_Generator(sz = num_nodes,temp = 10,temp_drop_frac = 0.9999).to(device)
            #gumbel_generator = gumbel_generator.double()
            optimizer_network = optim.Adam(gumbel_generator.parameters(),lr = 0.1)

            states_learner=Generator_states(batch*(batch_num+1),num_nodes).to(device)
            opt_states = optim.Adam(states_learner.parameters(), lr=0.1)


            #部分结构
            w1=torch.tensor(W1)
            w1=w1.float()
            w1=w1.to(device)


            w0=torch.tensor(W0).to(device)
            w0=w0.float()
            w0=w0.to(device)
            #损失函数
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn1 = torch.nn.MSELoss()
            #总迭代次数
            Epoch_Num=10
            #动力学学习次数
            Dyn_Steps=20
            #网络学习次数
            Net_Steps=10
            Sta_Steps=30
            #批数



            loss = 0
            dyn_losses = []
            net_losses = []
            accu_record = []
            err_nets = []
            acc_nets = []
            # start training
            losses = []
            accuracies = []
            losses_in_gumbel = []
            losses_state=[]
            for epoch in range(Epoch_Num):
                print('epoch running:'+str(epoch)+' / '+str(Epoch_Num))
                print('use gumbel')
                adj = gumbel_generator.sample(hard=True)
                # 先训练dynamics

                # dyn_learner.train()
                print("\n***************Dyn Training******************")
                for i in range(Dyn_Steps):
                    for j in range(batch_num):
                        X3=X2.copy()
                        Y3=Y2.copy()
                        X3[s1[j],:]=0
                        Y3[s1[j],:]=0
                        data_train=torch.Tensor(X3[:,batch*j:batch*(j+1)].T)[:,:,None]
                        data_target=torch.Tensor(Y3[:,batch*j:batch*(j+1)].T)
                        t=torch.Tensor([i for i in range(j*batch,(j+1)*batch)]).long().to(device)
                        data_train=data_train.to(device)
                        data_target=data_target.to(device)

                        loss = train_batch_dyn3(w0,optimizer_dyn,states_learner,dyn_learner,adj,data_train,data_target,loss_fn,s2[j],s1[j],t)
                        record_loss = loss.data.tolist()
                    losses.append(record_loss)

                print("\n***************State Training******************")

            #利用部分结构+P计算动力学acc
            pp=gumbel_generator.sample()
            acc3=0
            z=0
            t=torch.Tensor([i for i in range(batch_num*batch,(batch_num+1)*batch)]).long().to(device)
            unstates=states_learner(t)
            X3=X2.copy()
            Y3=Y2.copy()
            data_train=torch.Tensor(X3[:,batch_num*batch:(batch_num+1)*batch].T)[:,:,None]
            data_target=torch.Tensor(Y3[:,batch_num*batch:(batch_num+1)*batch].T)
            data_train=data_train.to(device)
            data_target=data_target.to(device)
            #data_train[:,s1[0],:]=dingxiao[:,s1[0],:]

            print('###########################')
            for t in range(0,100):
                z=z+1
                x_real=data_train[t,:,:][None,:,:]
                y_real=data_target[t,:][None,:].T

                y_real=y_real.view(1, -1)[0].long()
                adj=w1
                adj = adj.unsqueeze(0)
                adj = adj.repeat(1, 1, 1)
                x_real=x_real.to(device)        
                y_pred_1 = dyn_learner(x_real, adj)

                #print(y_pred.shape)
                y_pred_2=F.softmax(y_pred_1,2)

                p1=torch.argmax(y_pred_2, axis=2)
                t1=y_real

                t2=p1[:,:]
                for i in range(len(t1)):
                    if t2[:,i]==t1[i]:
                        acc3=acc3+1

            print("\n***************动力学预测ACC******************")
            print(acc3/(len(t1)*z))

            loss_state1=0
            loss_state11=0
            acc_state1=0
            for i in range(len(s1[0])):
                t=torch.Tensor([k for k in range(batch*batch_num,(batch*(batch_num+1)))]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,(batch*batch_num):(batch*(batch_num+1))]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state1=loss_state1+loss_fn2(dddd,ffff)
                loss_state11=loss_state11+loss_fn2(dddd,ffff)/(batch)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state1=acc_state1+1

            loss_state=0
            loss_state1=0
            acc_state=0
            for i in range(len(s1[0])):
                t=torch.Tensor([i for i in range(0,batch*batch_num)]).long().to(device)
                #plt.figure(figsize=(40,5))
                unstates=states_learner(t)
                fff=unstates[:,s1[0][i],:]
                ffff=np.squeeze(fff).to(device) 
                #plt.plot(ffff.detach().numpy())
                ddd=X2[:,0:batch*batch_num]
                dddd=torch.Tensor(np.squeeze(ddd[s1[0][i]])).to(device) 
                loss_fn2 = torch.nn.MSELoss(reduction='sum')
                loss_state=loss_state+loss_fn2(dddd,ffff)
                loss_state1=loss_state1+loss_fn2(dddd,ffff)/(batch*batch_num)

                ffff[ffff<0.5]=0
                ffff[ffff>=0.5]=1
                for i in range(ffff.shape[0]):
                    if ffff[i]==dddd[i]:
                        acc_state=acc_state+1
            print(loss_state)
            print(loss_state1)
            print(acc_state/(batch*batch_num*len(s1[0])))
            result1.append([acc3/(len(w1)*z),loss_state.item(),acc_state/(batch*batch_num*len(s1[0])),loss_state1.item(),acc_state1/(batch*len(s1[0]))])
        np.savetxt(net_path+'_'+dyn_name+"_del_ding"+str(int(del_net*100))+'_'+str(k)+".txt", np.array(result), fmt="%.6f", delimiter=" ")
        np.savetxt(net_path+'_'+dyn_name+"_del_xiao"+str(int(del_net*100))+'_'+str(k)+".txt", np.array(result1), fmt="%.6f", delimiter=" ")
    
