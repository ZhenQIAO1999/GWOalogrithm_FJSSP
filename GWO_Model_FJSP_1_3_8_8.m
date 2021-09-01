%该程序用于解决柔性作业车间调度，m个工件，n道工序，其中n为最大工序数，工件的工序
%数可以少于n，加工机器数为M，每个工件的每道工序具有多个机器可以选择，对应的时间
%不同，其中初始种群的储存方式采用cell数据类型
%Version:1.3
%fileDescription:调度机器可选的柔性作业车间问题，甘特图已完善,GWO,8*8实例
%last edit time:2019-6-7
function GWO_Model_FJSP_1_3_8_8()
count = 5000;     %迭代次数
N = 100;          %种群规模
m = 6;             %工件数
n = 4;             %工序数
M = 4;             %机器数
a =2;              %计算A/C协同系数的
plotif = 1;        %控制程序是否进行绘图
s = input(m,n);    %数据输入
[p,TN] = initial_p(m,n,N,s,M);    %生成初始种群50,采用细胞结构，每个元素为8*4
P = machine(n,M);
FIT = zeros(count,1);
aveFIT = zeros(count,1);
X1=randperm(count);       %收敛图形的横坐标X
X=sort(X1);
%------------------------输出最优解的时有用------------------------------
best_fit = 1000;            %改变模型需要修改此参数
best_p = zeros(m,n);
best_TN = zeros(m,n);
Y1p = zeros(m,1);
Y2p = zeros(m,1);
Y3p = zeros(m,1);
minfit3  =  1000000000;
%-------------------------进行迭代--------------------------------------
for i = 1:count
    [fit,Y1,Y2,Y3] = object(p,TN,N,P,m,n);   
    [newp,newTN] = GWO(fit,p,TN,N,m,n,s,a);
    a = a-2/(count-1);        %a的值会线性下降
    if best_fit > min(fit)
        [best_p,best_TN,best_fit,Y1p,Y2p,Y3p]=best(best_fit,best_p,fit,best_TN,Y1p,Y2p,Y3p,p,TN,Y1,Y2,Y3);
    end
    p = newp;
    TN = newTN;
    minfit = min(fit);
    if minfit3>minfit
        minfit3 = minfit;
    end
    FIT(i) = minfit3;    %用于适应度函数的
    aveFIT(i) = mean(fit);      %用于适应度函数的
end
%------------------投射最佳方案数据--------------------------------------
   
    fprintf('最优解：%d\n',best_fit);
    fprintf('工序1 工序2 工序3 工序4\n');
    best_p
    fprintf('时间1 时间2 时间3 时间4\n');
    best_TN
%------------------------收敛曲线----------------------------------------
    if plotif == 1
    figure;
    plot(X,FIT,'r');
    hold on;
    plot(X,aveFIT,'b');
    title('convergence curve');
    hold on;
    legend('optimal solution','average value');
%-------------------------甘特图-----------------------------------------
figure;
w=0.5;       %横条宽度 
set(gcf,'color','w');      %图的背景设为白色
for i = 1:m
    for j = 1:n
        color=[1,0.98,0.98;1,0.89,0.71;0.86,0.86,0.86;0.38,0.72,1;1,0,1;0,1,1;0,1,0.49;1,0.87,0.67;0.39,0.58,0.92;0.56,0.73,0.56];
        a = [Y1p(i,j),Y2p(i,j)];
        x=a(1,[1 1 2 2]);      %设置小图框四个点的x坐标
        y=Y3p(i,j)+[-w/2 w/2 w/2 -w/2];   %设置小图框四个点的y坐标
        color = [color(i,1),color(i,2),color(i,3)];
        p=patch('xdata',x,'ydata',y,'facecolor',color,'edgecolor','k');    %facecolor为填充颜色，edgecolor为图框颜色
            text(a(1,1)+0.5,Y3p(i,j),[num2str(i),'-',num2str(j)]);    %显示小图框里的数字位置和数值
    end
end
xlabel('process time/s');      %横坐标名称
ylabel('机器');            %纵坐标名称
title({[num2str(m),'*',num2str(M),' one of the optimal schedule（the makesoan is ',num2str(best_fit),')']});      %图形名称
axis([0,best_fit+2,0,M+1]);         %x轴，y轴的范围
set(gca,'Box','on');       %显示图形边框
set(gca,'YTick',0:M+1);     %y轴的增长幅度
set(gca,'YTickLabel',{'';num2str((1:M)','M%d');''});  %显示机器号
hold on;
    end
%--------------------------输入数据---------------------------------
function s = input(m,n)      %输入数据
s = cell(m,n);
s{1,1}=[1 2 3 4;100 80 110 120];
s{1,2}=[1 2 3 4;60 80 90 70];
s{1,3}=[1 2 3 4;90 60 80 100];
s{1,4}=[1 2 3 4;25 10 10 25];
s{2,1}=[1 2 3 4;100 80 110 120];
s{2,2}=[1 2 3 4;60 80 90 70];
s{2,3}=[1 2 3 4;90 60 80 100];
s{2,4}=[1 2 3 4;25 10 10 25];
s{3,1}=[1 2 3 4;100 80 110 120];
s{3,2}=[1 2 3 4;60 80 90 70];
s{3,3}=[1 2 3 4;90 60 80 100];
s{3,4}=[1 2 3 4;25 10 10 25];
s{4,1}=[1 2 3 4;100 80 110 120];
s{4,2}=[1 2 3 4;60 80 90 70];
s{4,3}=[1 2 3 4;90 60 80 100];
s{4,4}=[1 2 3 4;25 10 10 25];
s{5,1}=[1 2 3 4;100 80 110 120];
s{5,2}=[1 2 3 4;60 80 90 70];
s{5,3}=[1 2 3 4;90 60 80 100];
s{5,4}=[1 2 3 4;25 10 10 25];
s{6,1}=[1 2 3 4;100 80 110 120];
s{6,2}=[1 2 3 4;60 80 90 70];
s{6,3}=[1 2 3 4;90 60 80 100];
s{6,4}=[1 2 3 4;25 10 10 25];

%---------------------------建立初始种群-----------------------------
function [p,TN] = initial_p(m,n,N,s,M)     %建立初始种群
p = cell(N,1);            %p为初始解集的机器集
TN = cell(N,1);            %TN为初始解集的时间集
for i = 1:N                  %产生N个初始解
    store_m = zeros(M,1);    %用于储存生成初始方案时的各机器数量
    pz = zeros(m,n);         %pz为中间储存量，用于储存解i的机器号，大小为m*n
    tz = zeros(m,n);         %tz为中间储存量，用于储存解i的加工时间，大小为m*n
    for j = 1:m
        for k = 1:n
            sle = s(j,k);       %sle为工件j的工序k的数据，第一行为可选机器数，第二行为对应的加工时间
            sle2 = cell2mat(sle);    %sle为cell结构，需要将sle用cell2mat函数转换为double类型
            b = size(sle2,2);       %数据中有0数组，所以需要判断
            if b == 0
                pz(j,k) = 0;
                tz(j,k) = 0;
            else
            c = randperm(b,1);   %产生一个1到b的随机数，用于选择机器
                if store_m(c) >= (m*n)/M
                    c = randperm(b,1);
                        if store_m(c) >= (m*n)/M
                             c = randperm(b,1);
                             if store_m(c) >= (m*n)/M
                                c = randperm(b,1);
                             end
                        end
                end
            store_m(c) = store_m(c)+1;
            pz(j,k) = sle2(1,c);     %将机器赋予pz(j,k)
            tz(j,k) = sle2(2,c);     %将加工时间赋予tz(j,k)
            end
        end
    end
    p{i} = pz;
    TN{i} = tz;
end
%---------------------------输入各工序机器数量-----------------------
function P = machine(n,M)
P = zeros(n,1);
for i= 1:n
    P(i) = M;      %每道工序的可选机器数设为M
end
%-------------------------计算各染色体的适应度-----------------------
function [fit,Y1,Y2,Y3] = object(p,TN,N,P,m,n)  %计算各染色体的适应度
fit = zeros(N,1);
Y1 = cell(N,1);
Y2 = cell(N,1);
Y3 = cell(N,1);
    for j = 1:N
        Y1{j} = zeros(m,n);
        Y2{j} = zeros(m,n);
        Y3{j} = zeros(m,n);
    end
for w = 1:N
    X = p{w};                  %变量初始化
    T = TN{w};
    [m,n] = size(X);
    Y1p = zeros(m,n);
    Y2p = zeros(m,n);
    Y3p = zeros(m,n);
    Q1 = zeros(m,1);         %计算第一道工序的安排
    Q2 = zeros(m,1);
    R = X(:,1);             %取出第一道工序的机器号
    Q3 = floor(R);          %向下取整得到各工件在第一道工序使用的机器号
    for k =1:P(1)           %第一道工序的时间安排，k为机器号
        pos = find(Q3 == k);     %在Q3中取出用机器k加工的工件编号
        lenpos = length(pos);    %使用机器k的工件数量
        if lenpos == 0
        end
        if lenpos >= 1
            Q1(pos(1)) = 0;
            Q2(pos(1)) = Q1(pos(1)) + T(pos(1),1);
            if lenpos >= 2 
                for j = 2:lenpos
                    Q1(pos(j)) = Q2(pos(j-1));
                    Q2(pos(j)) = Q1(pos(j)) + T(pos(j),1);
                end
            end
        end
    end

    Y1p(:,1) = Q1;
    Y2p(:,1) = Q2;
    Y3p(:,1) = Q3;

    for k = 2:n            %计算第2到n道工序的安排
        Q1 = zeros(m,1);
        Q2 = zeros(m,1);
        R = X(:,k);        %取出第k道工序的机器号
        Q3 = floor(R);     %向下取整得到各工件在第k道工序使用的机器号
        R1 = X(:,k-1);     %取出前一道工序的机器号
        Q31 = floor(R1);   %向下取整得到各工件在前一道工序使用的机器号
        for i = 1:P(k)     %第i道工序的时间安排，k为机器号
            pos = find(Q3 == i);
            lenpos = length(pos);
            pos1 = find(Q31 == i);
            lenpos1 = length(pos1);
            if lenpos == 0
            end
            if lenpos >= 1
                EndTime = Y2p(pos(1),k-1);
                POS = zeros(1,lenpos1);
                for j = 1:lenpos1
                    POS(j) = Y2p(pos1(j),k-1);
                end
                EndTime1 = max(POS);
                if EndTime1 > EndTime
                    EndTime = EndTime1;
                else
                    EndTime = EndTime;
                end
                Q1(pos(1)) = EndTime;
                Q2(pos(1)) =  Q1(pos(1)) + T(pos(1),k-1);
                if lenpos >= 2
                    for j = 2:lenpos
                        Q1(pos(j)) = Y2p(pos(j),k-1);   %前一道工序的结束时间
                        if Q1(pos(j)) < Q2(pos(j-1))
                            Q1(pos(j)) = Q2(pos(j-1));
                        else
                             Q1(pos(j)) = Q1(pos(j));
                        end
                        Q2(pos(j)) = Q1(pos(j)) + T(pos(j),k);
                    end
                end
            end
        end
    Y1p(:,k) = Q1;
    Y2p(:,k) = Q2;
    Y3p(:,k) = Q3;
    end
    Y2m = Y2p(:,n);
    Y2m1 = Y2p(:,n-1);
    Zx = max(Y2m1);
    Zp = max(Y2m);
    if Zx >Zp
        Zp = Zx;
    end
    fit(w) = Zp;
    Y1{w} = Y1p;
    Y2{w} = Y2p;
    Y3{w} = Y3p;
end
%-----------------------------进行GWO操作---------------------------
function [newp,newTN] = GWO(fit,p,TN,N,m,n,s,a)
newp = cell(N,1);
newTN = cell(N,1);
fit_sort = sort(fit);
fit_1 = fit_sort(1);      %fit_1,fit_2,fit_3表示狼群中适应度最好的三个解
fit_2 = fit_sort(1);
fit_3 = fit_sort(1);
pos = find(fit == fit_1);
 if length(pos)>=3;      %判断最好的前三个解是否有重复，并放入p1,p2,p3
    p1 = p{pos(1)};
    p2 = p{pos(2)};
    p3 = p{pos(3)};
 elseif length(pos) == 2
     p1 = p{pos(1)};
     p2 = p{pos(2)};
     pos1 = find(fit==fit_3);
     p3 = p{pos1(1)};
 elseif length(pos) == 1 && length(find(fit ==fit_2))>=2
     p1 = p{pos(1)};
     pos2 = find(fit ==fit_2);
     p2 = p{pos2(1)};
     p3 = p{pos2(2)};
 elseif length(pos) == 1 && length(find(fit ==fit_2))==1
     p1 = p{pos(1)};
     p2 = p{find(fit ==fit_2)};
     pos3 = find(fit == fit_3);
     p3 = p{pos3(1)};
 end

for i=1:N          %对N个染色体进行更新
     p5=p1;        %为防止p1，p2，p3的数值发生变化，赋值给p5，p6，p7
     p6=p2;
     p7=p3;
    if i == 1     %将前三个最好的解放入下一代数的前三个解，保证最优解不会变坏
        newp{1} = p5;   
    elseif i==2
        newp{2}= p6;
    elseif i==3
        newp{3} = p7;
    elseif i>3
        p4 = p{i-3};   %将其余的灰狼取出
        for ii=1:m     %进行位置信息转换
            for j =1:n
                if length(find(s{ii,j}(1,:)==p5(ii,j)))>1
                    p5(ii,j) = 0;
                else
                    p5(ii,j) = find(s{ii,j}(1,:)==p5(ii,j));
                end
                if length(find(s{ii,j}(1,:)==p6(ii,j)))>1
                    p6(ii,j) = 0;
                else
                    p6(ii,j) = find(s{ii,j}(1,:)==p6(ii,j));
                end
                if length(find(s{ii,j}(1,:)==p7(ii,j)))>1
                    p7(ii,j) = 0;
                else
                    p7(ii,j) = find(s{ii,j}(1,:)==p7(ii,j));
                end
                if length(find(s{ii,j}(1,:)==p4(ii,j)))>1
                    p4(ii,j) = 0;
                else
                    p4(ii,j) = find(s{ii,j}(1,:)==p4(ii,j));
                end
            end
        end
        p4 = ceil((p5-(2*a*rand-a).*((2*rand).*p5-p4)+p6-(2*a*rand-a).*((2*rand).*p6-p4)+p7-(2*a*rand-a).*((2*rand).*p7-p4))/3);
        %灰狼位置更新公式
        for ii = 1:m         %将位置信息转换回机器号
            for j = 1:n
                if length(s{ii,j}(1,:))>p4(ii,j) || length(s{ii,j}(1,:))<p4(ii,j)
                    p4(ii,j) = s{ii,j}(1,ceil(rand*length(s{ii,j}(1,:))));
                else
                    p4(ii,j) = s{ii,j}(1,p4(ii,j));
                end
            end
        end
        newp{i} = p4;
    end
    for ii = 1:m     %对机器号对应的时间进行更新
        for j= 1:n
            pos = find(s{ii,j}(1,:)==newp{i}(ii,j));
            p8(ii,j) = s{ii,j}(2,pos(1));
        end
    end
    newTN{i} = p8;
end
%-----------------------------选择最优方案---------------------------
function [best_p,best_TN,best_fit,Y1p,Y2p,Y3p]=best(best_fit,best_p,fit,best_TN,Y1p,Y2p,Y3p,p,TN,Y1,Y2,Y3)
    best_fit = min(fit);
    pos = find(fit==best_fit);
    best_p = p(pos(1));
    best_TN = TN(pos(1));
    best_p=cell2mat(best_p);
    best_TN=cell2mat(best_TN);
    Y1p=Y1(pos(1));
    Y2p=Y2(pos(1));
    Y3p=Y3(pos(1));
    Y1p=cell2mat(Y1p);
    Y2p=cell2mat(Y2p);
    Y3p=cell2mat(Y3p);



