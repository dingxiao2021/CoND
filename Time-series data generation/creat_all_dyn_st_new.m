clear
%function creat_all_dyn_st
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%�������ж���ѧ���򣬱��� name_����ѧ_s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  �����������������  %%%%%%%%%%%%%
name=strvcat('BA_N200_m2');
T=5000;
type_network=0;      %�������� 0 Ϊ �б���ʽ������Ϊ�ڽӾ�����ʽ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dyn_name=strvcat('voter','kirman','ising','sis','game','language','threshold','majority'); %���ֶ���ѧ����

%%%%%%%%%%%%������
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
    load(strcat(name,'_w'))       %�ڽӾ���������ʽ name_w
end

t=11;         % ʱ�䲽
for i=1:8                %����8�ֶ���ѧ����
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





