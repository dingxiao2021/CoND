clear
%function creat_all_dyn_st
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%�������ж���ѧ���򣬱��� name_����ѧ_s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  �����������������  %%%%%%%%%%%%%
%name=strvcat('C:\Users\Administrator\Desktop\����\���������и����Ĵ��룬���ֽṹ���ֶ���ѧ\data\BA\BA_N100_m2');
%name=strvcat('C:\Users\Administrator\Desktop\���������\����\bio-grid-mouse');
name=strvcat('C:\Users\Administrator\Desktop\����\�ڵ���\WS_N500_k4_p05');%BA_N100_m2  ER_N100_p005 ER_N500_p001 WS_N100_k4_p05
%netname='bio-grid-mouse';
T=5000;
type_network=0;      %�������� 0 Ϊ �б���ʽ������Ϊ�ڽӾ�����ʽ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dyn_name=strvcat('voter','kirman','ising','sis','game','language','threshold','majority'); %���ֶ���ѧ����

%%%%%%%%%%%%������
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
    load(strcat(name,'_w'))       %�ڽӾ���������ʽ name_w
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


for i=[1,3,7]                %����8�ֶ���ѧ����
    i    
    [S,up]=fun_dyn_s(w,i,T);    
    up    
    tick=fix(T/up)  % ����
    tick_r=rem(T,up);  % ������
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
% for i=3                 %����8�ֶ���ѧ����
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





