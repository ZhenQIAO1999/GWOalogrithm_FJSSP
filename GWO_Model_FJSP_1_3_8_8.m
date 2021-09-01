%�ó������ڽ��������ҵ������ȣ�m��������n����������nΪ��������������Ĺ���
%����������n���ӹ�������ΪM��ÿ��������ÿ��������ж����������ѡ�񣬶�Ӧ��ʱ��
%��ͬ�����г�ʼ��Ⱥ�Ĵ��淽ʽ����cell��������
%Version:1.3
%fileDescription:���Ȼ�����ѡ��������ҵ�������⣬����ͼ������,GWO,8*8ʵ��
%last edit time:2019-6-7
function GWO_Model_FJSP_1_3_8_8()
count = 5000;     %��������
N = 100;          %��Ⱥ��ģ
m = 6;             %������
n = 4;             %������
M = 4;             %������
a =2;              %����A/CЭͬϵ����
plotif = 1;        %���Ƴ����Ƿ���л�ͼ
s = input(m,n);    %��������
[p,TN] = initial_p(m,n,N,s,M);    %���ɳ�ʼ��Ⱥ50,����ϸ���ṹ��ÿ��Ԫ��Ϊ8*4
P = machine(n,M);
FIT = zeros(count,1);
aveFIT = zeros(count,1);
X1=randperm(count);       %����ͼ�εĺ�����X
X=sort(X1);
%------------------------������Ž��ʱ����------------------------------
best_fit = 1000;            %�ı�ģ����Ҫ�޸Ĵ˲���
best_p = zeros(m,n);
best_TN = zeros(m,n);
Y1p = zeros(m,1);
Y2p = zeros(m,1);
Y3p = zeros(m,1);
minfit3  =  1000000000;
%-------------------------���е���--------------------------------------
for i = 1:count
    [fit,Y1,Y2,Y3] = object(p,TN,N,P,m,n);   
    [newp,newTN] = GWO(fit,p,TN,N,m,n,s,a);
    a = a-2/(count-1);        %a��ֵ�������½�
    if best_fit > min(fit)
        [best_p,best_TN,best_fit,Y1p,Y2p,Y3p]=best(best_fit,best_p,fit,best_TN,Y1p,Y2p,Y3p,p,TN,Y1,Y2,Y3);
    end
    p = newp;
    TN = newTN;
    minfit = min(fit);
    if minfit3>minfit
        minfit3 = minfit;
    end
    FIT(i) = minfit3;    %������Ӧ�Ⱥ�����
    aveFIT(i) = mean(fit);      %������Ӧ�Ⱥ�����
end
%------------------Ͷ����ѷ�������--------------------------------------
   
    fprintf('���Ž⣺%d\n',best_fit);
    fprintf('����1 ����2 ����3 ����4\n');
    best_p
    fprintf('ʱ��1 ʱ��2 ʱ��3 ʱ��4\n');
    best_TN
%------------------------��������----------------------------------------
    if plotif == 1
    figure;
    plot(X,FIT,'r');
    hold on;
    plot(X,aveFIT,'b');
    title('convergence curve');
    hold on;
    legend('optimal solution','average value');
%-------------------------����ͼ-----------------------------------------
figure;
w=0.5;       %������� 
set(gcf,'color','w');      %ͼ�ı�����Ϊ��ɫ
for i = 1:m
    for j = 1:n
        color=[1,0.98,0.98;1,0.89,0.71;0.86,0.86,0.86;0.38,0.72,1;1,0,1;0,1,1;0,1,0.49;1,0.87,0.67;0.39,0.58,0.92;0.56,0.73,0.56];
        a = [Y1p(i,j),Y2p(i,j)];
        x=a(1,[1 1 2 2]);      %����Сͼ���ĸ����x����
        y=Y3p(i,j)+[-w/2 w/2 w/2 -w/2];   %����Сͼ���ĸ����y����
        color = [color(i,1),color(i,2),color(i,3)];
        p=patch('xdata',x,'ydata',y,'facecolor',color,'edgecolor','k');    %facecolorΪ�����ɫ��edgecolorΪͼ����ɫ
            text(a(1,1)+0.5,Y3p(i,j),[num2str(i),'-',num2str(j)]);    %��ʾСͼ���������λ�ú���ֵ
    end
end
xlabel('process time/s');      %����������
ylabel('����');            %����������
title({[num2str(m),'*',num2str(M),' one of the optimal schedule��the makesoan is ',num2str(best_fit),')']});      %ͼ������
axis([0,best_fit+2,0,M+1]);         %x�ᣬy��ķ�Χ
set(gca,'Box','on');       %��ʾͼ�α߿�
set(gca,'YTick',0:M+1);     %y�����������
set(gca,'YTickLabel',{'';num2str((1:M)','M%d');''});  %��ʾ������
hold on;
    end
%--------------------------��������---------------------------------
function s = input(m,n)      %��������
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

%---------------------------������ʼ��Ⱥ-----------------------------
function [p,TN] = initial_p(m,n,N,s,M)     %������ʼ��Ⱥ
p = cell(N,1);            %pΪ��ʼ�⼯�Ļ�����
TN = cell(N,1);            %TNΪ��ʼ�⼯��ʱ�伯
for i = 1:N                  %����N����ʼ��
    store_m = zeros(M,1);    %���ڴ������ɳ�ʼ����ʱ�ĸ���������
    pz = zeros(m,n);         %pzΪ�м䴢���������ڴ����i�Ļ����ţ���СΪm*n
    tz = zeros(m,n);         %tzΪ�м䴢���������ڴ����i�ļӹ�ʱ�䣬��СΪm*n
    for j = 1:m
        for k = 1:n
            sle = s(j,k);       %sleΪ����j�Ĺ���k�����ݣ���һ��Ϊ��ѡ���������ڶ���Ϊ��Ӧ�ļӹ�ʱ��
            sle2 = cell2mat(sle);    %sleΪcell�ṹ����Ҫ��sle��cell2mat����ת��Ϊdouble����
            b = size(sle2,2);       %��������0���飬������Ҫ�ж�
            if b == 0
                pz(j,k) = 0;
                tz(j,k) = 0;
            else
            c = randperm(b,1);   %����һ��1��b�������������ѡ�����
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
            pz(j,k) = sle2(1,c);     %����������pz(j,k)
            tz(j,k) = sle2(2,c);     %���ӹ�ʱ�丳��tz(j,k)
            end
        end
    end
    p{i} = pz;
    TN{i} = tz;
end
%---------------------------����������������-----------------------
function P = machine(n,M)
P = zeros(n,1);
for i= 1:n
    P(i) = M;      %ÿ������Ŀ�ѡ��������ΪM
end
%-------------------------�����Ⱦɫ�����Ӧ��-----------------------
function [fit,Y1,Y2,Y3] = object(p,TN,N,P,m,n)  %�����Ⱦɫ�����Ӧ��
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
    X = p{w};                  %������ʼ��
    T = TN{w};
    [m,n] = size(X);
    Y1p = zeros(m,n);
    Y2p = zeros(m,n);
    Y3p = zeros(m,n);
    Q1 = zeros(m,1);         %�����һ������İ���
    Q2 = zeros(m,1);
    R = X(:,1);             %ȡ����һ������Ļ�����
    Q3 = floor(R);          %����ȡ���õ��������ڵ�һ������ʹ�õĻ�����
    for k =1:P(1)           %��һ�������ʱ�䰲�ţ�kΪ������
        pos = find(Q3 == k);     %��Q3��ȡ���û���k�ӹ��Ĺ������
        lenpos = length(pos);    %ʹ�û���k�Ĺ�������
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

    for k = 2:n            %�����2��n������İ���
        Q1 = zeros(m,1);
        Q2 = zeros(m,1);
        R = X(:,k);        %ȡ����k������Ļ�����
        Q3 = floor(R);     %����ȡ���õ��������ڵ�k������ʹ�õĻ�����
        R1 = X(:,k-1);     %ȡ��ǰһ������Ļ�����
        Q31 = floor(R1);   %����ȡ���õ���������ǰһ������ʹ�õĻ�����
        for i = 1:P(k)     %��i�������ʱ�䰲�ţ�kΪ������
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
                        Q1(pos(j)) = Y2p(pos(j),k-1);   %ǰһ������Ľ���ʱ��
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
%-----------------------------����GWO����---------------------------
function [newp,newTN] = GWO(fit,p,TN,N,m,n,s,a)
newp = cell(N,1);
newTN = cell(N,1);
fit_sort = sort(fit);
fit_1 = fit_sort(1);      %fit_1,fit_2,fit_3��ʾ��Ⱥ����Ӧ����õ�������
fit_2 = fit_sort(1);
fit_3 = fit_sort(1);
pos = find(fit == fit_1);
 if length(pos)>=3;      %�ж���õ�ǰ�������Ƿ����ظ���������p1,p2,p3
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

for i=1:N          %��N��Ⱦɫ����и���
     p5=p1;        %Ϊ��ֹp1��p2��p3����ֵ�����仯����ֵ��p5��p6��p7
     p6=p2;
     p7=p3;
    if i == 1     %��ǰ������õĽ������һ������ǰ�����⣬��֤���Žⲻ��仵
        newp{1} = p5;   
    elseif i==2
        newp{2}= p6;
    elseif i==3
        newp{3} = p7;
    elseif i>3
        p4 = p{i-3};   %������Ļ���ȡ��
        for ii=1:m     %����λ����Ϣת��
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
        %����λ�ø��¹�ʽ
        for ii = 1:m         %��λ����Ϣת���ػ�����
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
    for ii = 1:m     %�Ի����Ŷ�Ӧ��ʱ����и���
        for j= 1:n
            pos = find(s{ii,j}(1,:)==newp{i}(ii,j));
            p8(ii,j) = s{ii,j}(2,pos(1));
        end
    end
    newTN{i} = p8;
end
%-----------------------------ѡ�����ŷ���---------------------------
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



