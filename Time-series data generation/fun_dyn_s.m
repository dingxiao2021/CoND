function [S,up]=fun_dyn_s(w,id,T)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%各种动力学生成S
%输入：网络w，生成的时间序列个数T
%动力学标号：id=
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
            disp('动力学：Voter');
            up=11;        %每隔多少步重新初始化
            S=fun_create_voter_s(w,T,up);            
        case 2
            disp('动力学：Kirman');
            c1=0.008;
            c2=0.008;
            d=0.08;
            S=fun_create_kirman_s(w,c1,c2,d,T);
        case 3
            disp('动力学：Ising');
            bata=2;
            S=fun_create_ising_s(w,bata,T);      
        case 4
            disp('动力学：SIS');
            alpha=0.2;
            bata=0.1;
            S=fun_create_sis_s(w,alpha,bata,T); 
        case 5
             disp('动力学：Game');
             a=5;
             b=10;
             S=fun_create_game_s(w,a,b,T);       
        case 6
             disp('动力学：language');
             up=50;        %每隔多少步重新初始化
             S=fun_create_language_s(w,T,up);      
        case 7
             disp('动力学：Threshold');
             up=5;        %每隔多少步重新初始化  
             S=fun_create_determine_s(w,T,up);
        case 8
             disp('动力学：majority');
             up=5;        %每隔多少步重新初始化 
             S=fun_create_majority_s(w,T,up);
        otherwise
            disp('提示：请输入1—8');
    end
end

function  S=fun_create_voter_s(w,T,up)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %voter动力学生成时间序列S
    %输入 w 网络矩阵
    %输入 T 产生的序列长度
    %输入 up 每多少步重新初始化
    %输出 序列 S

    %%%%需要调用     RUN_vote.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %节点个数
    % T=15000;      %生成序列
    % up=50;        %每隔多少步重新初始化
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %节点个数

    %构造网的列表形式（c调用）
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %执行C（混编）程序 输出strategy.txt
    RUN_vote(N,T,up)
    delete nb.txt

    %保存成S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_kirman_s(w,c1,c2,d,T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %kirman动力学生成时间序列S
    %输入 w 网络矩阵
    %输入 参数 c1 c2 d c1+dm概率感染 c2+d(k-m)概率恢复
    %输入 T 产生的序列长度

    %输出 序列 S

    %%%%需要调用     RUN_kirman.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %节点个数
    %c1=0.1;
    %c2=0.1;
    %d=0.08;
    % T=15000;      %生成序列

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %节点个数

    %构造网的列表形式（c调用）
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %执行C（混编）程序 输出strategy.txt
     RUN_kirman(c1,c2,d,N,T)
    delete nb.txt

    %保存成S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end



function  S=fun_create_ising_s(w,bata,T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SiS动力学生成时间序列S
    %输入 w 网络矩阵
    %输入 bata   参数
    %输入 T 产生的序列长度
    %输出 序列 S

    %%%%需要调用     RUN_ising.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % T=15000;

    % bata=2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %节点个数

    %构造网的列表形式（c调用）
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %执行C（混编）程序 输出strategy.txt
    RUN_ising(bata,N,T)
    delete nb.txt

    %保存成S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end


function  S=fun_create_sis_s(w,alpha,bata,T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SiS动力学生成时间序列S
    %输入 w 网络矩阵
    %输入 alpha  传播率
    %输入 bata   回复率
    %输入 T 产生的序列长度
    %输入 up 每多少步重新初始化
    %输出 序列 S

    %%%%需要调用     RUN_sis.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % T=15000;
    % alpha=0.1
    % bata=0.5
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %节点个数

    %构造网的列表形式（c调用）
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %执行C（混编）程序 输出strategy.txt
    RUN_sis(alpha,bata,N,T)
    delete nb.txt

    %保存成S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_game_s(w,a,b,T)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SiS动力学生成时间序列S
    %输入 w 网络矩阵
    %输入 a b   参数[a,b;0,0]
    %输入 T 产生的序列长度
    %输出 序列 S

    %%%%需要调用     RUN_game.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % T=15000;
    %a=5
    %b=5

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %节点个数

    %构造网的列表形式（c调用）
    [a1,b1]=find(w);
    nb=[b1,a1]-1;
    dlmwrite('nb.txt',nb,' ');

    %执行C（混编）程序 输出strategy.txt
    RUN_game(a,b,N,T)
    delete nb.txt

    %保存成S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_language_s(w,T,up)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %language动力学生成时间序列S
    %输入 w 网络矩阵
    %输入 T 产生的序列长度
    %输入 up 每多少步重新初始化
    %输出 序列 S

    %%%%需要调用     RUN_language.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %节点个数
    % T=15000;      %生成序列
    % up=15000;        %每隔多少步重新初始化
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %节点个数

    %构造网的列表形式（c调用）
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %执行C（混编）程序 输出strategy.txt
    RUN_language(N,T,up)
    delete nb.txt

    %保存成S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_determine_s(w,T,up)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %determine(阈值)动力学生成时间序列S
    %输入 w 网络矩阵
    %输入 T 产生的序列长度
    %输入 up 每多少步重新初始化
    %输出 序列 S

    %%%%需要调用     RUN_determine.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %节点个数
    % T=15000;      %生成序列
    % up=10;        %每隔多少步重新初始化
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %节点个数

    %构造网的列表形式（c调用）
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %执行C（混编）程序 输出strategy.txt
    RUN_determine(N,T,up)
    delete nb.txt

    %保存成S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end

function  S=fun_create_majority_s(w,T,up)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %majority动力学生成时间序列S
    %输入 w 网络矩阵
    %输入 T 产生的序列长度
    %输入 up 每多少步重新初始化
    %输出 序列 S

    %%%%需要调用     RUN_majority.mexw64
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % N=length(w);  %节点个数
    % T=15000;      %生成序列
    % up=100;        %每隔多少步重新初始化
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=length(w);  %节点个数

    %构造网的列表形式（c调用）
    [a,b]=find(w);
    nb=[b,a]-1;
    dlmwrite('nb.txt',nb,' ');

    %执行C（混编）程序 输出strategy.txt
    RUN_majority(N,T,up)
    delete nb.txt

    %保存成S
    load strategy.txt
    S=strategy;
    delete strategy.txt
end
