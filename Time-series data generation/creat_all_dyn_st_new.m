clear
%function creat_all_dyn_st
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%生成所有动力学程序，保存 name_动力学_s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  网络名字与迭代次数  %%%%%%%%%%%%%
name=strvcat('BA_N200_m2');
T=5000;
type_network=0;      %网络类型 0 为 列表形式，其他为邻接矩阵形式
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dyn_name=strvcat('voter','kirman','ising','sis','game','language','threshold','majority'); %八种动力学名字

%%%%%%%%%%%%读网络
% eval('1+1')
if type_network==0
   a=load(strcat(name,'.txt'));
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

t=11;         % 时间步
for i=1:8                %生成8种动力学程序
    dyn_name(i,:)
    x=[];
    y=[];
    for j=1:1000 
        [S,up]=fun_dyn_s(w,i,t); 
        x=[x;S(1:t-1,:)];
        y=[y;S(2:t,:)];
    end    
    dlmwrite([name,'_',strtrim(dyn_name(i,:)),'_x.txt'],x,'delimiter',' ');
    dlmwrite([name,'_',strtrim(dyn_name(i,:)),'_y.txt'],y,'delimiter',' ');
end





