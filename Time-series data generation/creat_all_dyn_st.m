clear
%function creat_all_dyn_st
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%生成所有动力学程序，保存 name_动力学_s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  网络名字与迭代次数  %%%%%%%%%%%%%
%name=strvcat('C:\Users\Administrator\Desktop\胡雯\在这里面有个核心代码，部分结构部分动力学\data\BA\BA_N100_m2');
%name=strvcat('C:\Users\Administrator\Desktop\复杂网络包\数据\bio-grid-mouse');
name=strvcat('C:\Users\Administrator\Desktop\胡雯\节点数\WS_N500_k4_p05');%BA_N100_m2  ER_N100_p005 ER_N500_p001 WS_N100_k4_p05
%netname='bio-grid-mouse';
T=5000;
type_network=0;      %网络类型 0 为 列表形式，其他为邻接矩阵形式
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dyn_name=strvcat('voter','kirman','ising','sis','game','language','threshold','majority'); %八种动力学名字

%%%%%%%%%%%%读网络
% eval('1+1')
if type_network==0
   a=load(strcat(name,'.txt'));
   %a=load(strcat(name,'.edges'));
    if ~all(all(a(:,1:2)))
        a(:,1:2)=a(:,1:2)+1;
    end
    n=max(max(a));
    a(:,3)=1;
    w=spconvert(a);
    w(n,n)=0;
    w=spones(w+w');    
else
    load(strcat(name,'_w'))       %邻接矩阵命名形式 name_w
end

% G=graph(w);
% [bin,binsize] = conncomp(G);
% idx = binsize(bin) == max(binsize);
% SG = subgraph(G, idx);
% w = full(adjacency(SG));

% n
% m=size(a,1)
% %m=size(SG.Edges,1)
% k=2*m/n


for i=[1,3,7]                %生成8种动力学程序
    i    
    [S,up]=fun_dyn_s(w,i,T);    
    up    
    tick=fix(T/up)  % 求商
    tick_r=rem(T,up);  % 求余数
    x=[];
    y=[];
    for j=1:tick
        x=[x;S(up*(j-1)+1:up*j-1,:)];
        y=[y;S(up*(j-1)+2:up*j,:)];
    end
    %eval(['save ',name,'_',strtrim(dyn_name(i,:)),'_x',' x'])
    %eval(['save ',name,'_',strtrim(dyn_name(i,:)),'_y',' y'])
    dlmwrite([name,'_',strtrim(dyn_name(i,:)),'_x.txt'],x,'delimiter',' ');
    dlmwrite([name,'_',strtrim(dyn_name(i,:)),'_y.txt'],y,'delimiter',' ');
    %dlmwrite(['C:\Users\Administrator\Desktop\dyn_net\net200\',netname,'_',strtrim(dyn_name(i,:)),'_x.txt'],x,'delimiter',' ');
    %dlmwrite(['C:\Users\Administrator\Desktop\dyn_net\net200\',netname,'_',strtrim(dyn_name(i,:)),'_y.txt'],y,'delimiter',' ');
end

% t=11;
% for i=3                 %生成8种动力学程序
%     dyn_name(i,:)
%     x=[];
%     y=[];
%     for j=1:1000 
%         j
%         [S,up]=fun_dyn_s(w,i,t); 
%         x=[x;S(1:t-1,:)];
%         y=[y;S(2:t,:)];
%     end    
%     dlmwrite([name,'_',strtrim(dyn_name(i,:)),'_x.txt'],x,'delimiter',' ');
%     dlmwrite([name,'_',strtrim(dyn_name(i,:)),'_y.txt'],y,'delimiter',' ');
% end





