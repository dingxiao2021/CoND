function [S,up]=fun_dyn_s(w,id,T)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%���ֶ���ѧ����S
%���룺����w�����ɵ�ʱ�����и���T
%����ѧ��ţ�id=
%id=1   Voter
%id=2   Kirman
%id=3   Ising
%id=4   SIS
%id=5   Game
%id=6   Languge
%id=7   Threhold
%id=8   Majority
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
up=T;

    switch id
        case 1
            disp('����ѧ��Voter');
            up=11;        %ÿ�����ٲ����³�ʼ��
            S=fun_create_voter_s(w,T,up);            
        case 2
            disp('����ѧ��Kirman');
            c1=0.008;
            c2=0.008;
            d=0.08;
            S=fun_create_kirman_s(w,c1,c2,d,T);
        case 3
            disp('����ѧ��Ising');
            bata=2;
            S=fun_create_ising_s(w,bata,T);      
        case 4
            disp('����ѧ��SIS');
            alpha=0.2;
            bata=0.1;
            S=fun_create_sis_s(w,alpha,bata,T); 
        case 5
             disp('����ѧ��Game');
             a=5;
             b=10;
             S=fun_create_game_s(w,a,b,T);       
        case 6
             disp('����ѧ��language');
             up=50;        %ÿ�����ٲ����³�ʼ��
             S=fun_create_language_s(w,T,up);      
        case 7
             disp('����ѧ��Threshold');
             up=5;        %ÿ�����ٲ����³�ʼ��  
             S=fun_create_determine_s(w,T,up);
        case 8
             disp('����ѧ��majority');
             up=5;        %ÿ�����ٲ����³�ʼ�� 
             S=fun_create_majority_s(w,T,up);
        otherwise
            disp('��ʾ��������1��8');
    end
end

function  S=fun_create_voter_s(w,T,up)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %voter����ѧ����ʱ������S
    %���� w �������
    %���� T ���������г���
    %���� up ÿ���ٲ����³�ʼ��
    %��� ���� S

    %%%%��Ҫ����     RUN_vote.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %�ڵ����
    % T=15000;      %��������
    % up=50;        %ÿ�����ٲ����³�ʼ��
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %�ڵ����

    %���������б���ʽ��c���ã�
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %ִ��C����ࣩ���� ���strategy.txt
    RUN_vote(N,T,up)
    delete nb.txt

    %�����S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_kirman_s(w,c1,c2,d,T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %kirman����ѧ����ʱ������S
    %���� w �������
    %���� ���� c1 c2 d c1+dm���ʸ�Ⱦ c2+d(k-m)���ʻָ�
    %���� T ���������г���

    %��� ���� S

    %%%%��Ҫ����     RUN_kirman.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %�ڵ����
    %c1=0.1;
    %c2=0.1;
    %d=0.08;
    % T=15000;      %��������

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %�ڵ����

    %���������б���ʽ��c���ã�
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %ִ��C����ࣩ���� ���strategy.txt
     RUN_kirman(c1,c2,d,N,T)
    delete nb.txt

    %�����S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end



function  S=fun_create_ising_s(w,bata,T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SiS����ѧ����ʱ������S
    %���� w �������
    %���� bata   ����
    %���� T ���������г���
    %��� ���� S

    %%%%��Ҫ����     RUN_ising.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % T=15000;

    % bata=2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %�ڵ����

    %���������б���ʽ��c���ã�
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %ִ��C����ࣩ���� ���strategy.txt
    RUN_ising(bata,N,T)
    delete nb.txt

    %�����S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end


function  S=fun_create_sis_s(w,alpha,bata,T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SiS����ѧ����ʱ������S
    %���� w �������
    %���� alpha  ������
    %���� bata   �ظ���
    %���� T ���������г���
    %���� up ÿ���ٲ����³�ʼ��
    %��� ���� S

    %%%%��Ҫ����     RUN_sis.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % T=15000;
    % alpha=0.1
    % bata=0.5
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %�ڵ����

    %���������б���ʽ��c���ã�
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %ִ��C����ࣩ���� ���strategy.txt
    RUN_sis(alpha,bata,N,T)
    delete nb.txt

    %�����S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_game_s(w,a,b,T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SiS����ѧ����ʱ������S
    %���� w �������
    %���� a b   ����[a,b;0,0]
    %���� T ���������г���
    %��� ���� S

    %%%%��Ҫ����     RUN_game.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % T=15000;
    %a=5
    %b=5

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %�ڵ����

    %���������б���ʽ��c���ã�
    [a1,b1]=find(w);
    nb=[b1,a1]-1;
    dlmwrite('nb.txt',nb,' ');

    %ִ��C����ࣩ���� ���strategy.txt
    RUN_game(a,b,N,T)
    delete nb.txt

    %�����S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_language_s(w,T,up)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %language����ѧ����ʱ������S
    %���� w �������
    %���� T ���������г���
    %���� up ÿ���ٲ����³�ʼ��
    %��� ���� S

    %%%%��Ҫ����     RUN_language.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %�ڵ����
    % T=15000;      %��������
    % up=15000;        %ÿ�����ٲ����³�ʼ��
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %�ڵ����

    %���������б���ʽ��c���ã�
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %ִ��C����ࣩ���� ���strategy.txt
    RUN_language(N,T,up)
    delete nb.txt

    %�����S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_determine_s(w,T,up)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %determine(��ֵ)����ѧ����ʱ������S
    %���� w �������
    %���� T ���������г���
    %���� up ÿ���ٲ����³�ʼ��
    %��� ���� S

    %%%%��Ҫ����     RUN_determine.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %�ڵ����
    % T=15000;      %��������
    % up=10;        %ÿ�����ٲ����³�ʼ��
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %�ڵ����

    %���������б���ʽ��c���ã�
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %ִ��C����ࣩ���� ���strategy.txt
    RUN_determine(N,T,up)
    delete nb.txt

    %�����S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_majority_s(w,T,up)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %majority����ѧ����ʱ������S
    %���� w �������
    %���� T ���������г���
    %���� up ÿ���ٲ����³�ʼ��
    %��� ���� S

    %%%%��Ҫ����     RUN_majority.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %�ڵ����
    % T=15000;      %��������
    % up=100;        %ÿ�����ٲ����³�ʼ��
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %�ڵ����

    %���������б���ʽ��c���ã�
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %ִ��C����ࣩ���� ���strategy.txt
    RUN_majority(N,T,up)
    delete nb.txt

    %�����S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end
