%%%%    %%%%%%load in data   %%%%040102, 080402, 040402
clear all;
close all;
%load('n040102.mat');
load('n080402.mat');

rng(1);

freq = 10;           %%%xx mins per datapoint

list = floor(numel(joindata.power_consumption)/(freq*2));

for i=1:list
    a = mean(joindata.dateTime((freq*2*(i-1))+1:(freq*2*i)));
    date_num(i) = mean(joindata.dateTime((freq*2*(i-1))+1:(freq*2*i)));
    date_str{i}= datestr(a);
    month_num(i)= month(a);
    week_num(i)= weeknum(a);
    power(i) = mean(joindata.power_consumption((freq*2*(i-1))+1:(freq*2*i)));
    power_status(i) = mean(joindata.power_status((freq*2*(i-1))+1:(freq*2*i)));
    do(i) = mean(joindata.desired_operation((freq*2*(i-1))+1:(freq*2*i)));
    dt(i) = mean(joindata.target_temperature((freq*2*(i-1))+1:(freq*2*i)));
    it(i) = mean(joindata.temperature((freq*2*(i-1))+1:(freq*2*i)));
    ot(i) = mean(joindata.outTempCon((freq*2*(i-1))+1:(freq*2*i)));
    %hi(i) = mean(joindata.heatindexCon((freq*2*(i-1))+1:(freq*2*i))); 
    vardo(i) = var(joindata.desired_operation((freq*2*(i-1))+1:(freq*2*i)));
    vardt(i) = var(joindata.target_temperature((freq*2*(i-1))+1:(freq*2*i)));
    varit(i) = var(joindata.temperature((freq*2*(i-1))+1:(freq*2*i)));
    varot(i) = var(joindata.outTempCon((freq*2*(i-1))+1:(freq*2*i)));
    dton(i) = dt(i)*power_status(i);
end

jointable = [array2table(date_num', 'VariableNames',{'date_num'}), cell2table(date_str','VariableNames',{'date_str'}),...
    array2table(month_num','VariableNames',{'month_num'}), array2table(week_num','VariableNames',{'week_num'}),...
    array2table(power','VariableNames',{'power'}), array2table(power_status','VariableNames',{'power_status'}),...
    array2table(do','VariableNames',{'do'}), array2table(dt','VariableNames',{'dt'}'), array2table(it','VariableNames',{'it'}),...
    array2table(ot','VariableNames',{'ot'}),array2table(dton','VariableNames',{'dton'})...
    array2table(vardo','VariableNames',{'vardo'}),array2table(vardt','VariableNames',{'vardt'}),array2table(varit','VariableNames',{'varit'}),...
    array2table(vardo','VariableNames',{'varot'})];

jointable.power_status(jointable.power_status<1)=0;

p1= [0;jointable.power(1:end-1)];

mavgot30min= sum(lagmatrix(jointable.ot,[1:(30/freq)]),2)/(30/freq);
mavgot1hr= sum(lagmatrix(jointable.ot,[1:(60/freq)]),2)/(60/freq);
mavgot2hr= sum(lagmatrix(jointable.ot,[1:(120/freq)]),2)/(120/freq);
mvarot30min=var(lagmatrix(jointable.ot,[1:(30/freq)])')';
mvarot1hr=var(lagmatrix(jointable.ot,[1:(60/freq)])')';
mvarot2hr=var(lagmatrix(jointable.ot,[1:(120/freq)])')';

mavgit30min= sum(lagmatrix(jointable.it,[1:(30/freq)]),2)/(30/freq);
mavgit1hr= sum(lagmatrix(jointable.it,[1:(60/freq)]),2)/(60/freq);
mavgit2hr= sum(lagmatrix(jointable.it,[1:(120/freq)]),2)/(120/freq);
mvarit30min=var(lagmatrix(jointable.it,[1:(30/freq)])')';
mvarit1hr=var(lagmatrix(jointable.it,[1:(60/freq)])')';
mvarit2hr=var(lagmatrix(jointable.it,[1:(120/freq)])')';

mavgdt30min= sum(lagmatrix(jointable.dt,[1:(30/freq)]),2)/(30/freq);
mavgdt1hr= sum(lagmatrix(jointable.dt,[1:(60/freq)]),2)/(60/freq);
mavgdt2hr= sum(lagmatrix(jointable.dt,[1:(120/freq)]),2)/(120/freq);
mvardt30min=var(lagmatrix(jointable.dt,[1:(30/freq)])')';
mvardt1hr=var(lagmatrix(jointable.dt,[1:(60/freq)])')';
mvardt2hr=var(lagmatrix(jointable.dt,[1:(120/freq)])')';

changeot30min = mavgot30min-[0;mavgot30min(1:end-1)];
changeit30min = mavgit30min-[0;mavgit30min(1:end-1)];
changedt30min = mavgdt30min-[0;mavgdt30min(1:end-1)];

%% finding time on since
st1 = [0;jointable.power_status(1:end-1)];
stchange = jointable.power_status-st1;
a=find((stchange>0));
b=find((stchange<0));            
tonsin = zeros([numel(jointable.power_status) 1]);
tlastoncounter = zeros([numel(jointable.power_status) 1]);
tlaston = zeros([numel(jointable.power_status) 1]);
for k=1:numel(b)
    tonsin(a(k):b(k))= [1:(b(k)-a(k)+1)]';
    if numel(a)>numel(b)
        tlastoncounter(b(k)+1:a(k+1))= [1:(a(k+1)-b(k))]';
        if k+1<numel(b)
            tlaston(a(k+1):b(k+1))=max([1:(a(k+1)-b(k))]);
        end
    end
end
tlastoncounter(tlastoncounter>144)=144;
tlaston(tlaston>144) = 144; 

%% time since change in dt
changedton = jointable.dton-[0;jointable.dton(1:end-1)];
counter =0;

for k =1:numel(changedton)
    if changedton(k)~=0
       tsindt(k) = 0;
       counter=0;
    else 
        counter=counter+1;
        tsindt(k) = counter;
    end
end

tsindt = tsindt';

ditemp30min = mavgdt30min-mavgit30min;

jointable = [jointable, array2table(p1), array2table(tonsin), array2table(tsindt), array2table(tlaston),array2table(tlastoncounter), array2table(mavgot30min) ,array2table(mavgot1hr), array2table(mavgot2hr),array2table(mvarot30min) ,array2table(mvarot1hr), array2table(mvarot2hr),...
    array2table(mavgit30min) ,array2table(mavgit1hr), array2table(mavgit2hr),array2table(mvarit30min) ,array2table(mvarit1hr), array2table(mvarit2hr),...
    array2table(mavgdt30min) ,array2table(mavgdt1hr), array2table(mavgdt2hr),array2table(mvardt30min) ,array2table(mvardt1hr), array2table(mvardt2hr),...
    array2table(changeot30min), array2table(changeit30min), array2table(changedt30min), array2table(ditemp30min)];

clearvars -except jointable

%%%%% training and test set selection

train = jointable(jointable.power_status==1&jointable.month_num>6&jointable.month_num<12,:);
test = [jointable(jointable.power_status==1&jointable.month_num==12,:);jointable(jointable.power_status==1&jointable.month_num==5,:)];

temptrainx = lagmatrix(table2array(jointable(jointable.month_num>6&jointable.month_num<12,[5:8,10:11,17:42])),1);
temptrainnames = strcat(jointable(jointable.month_num>6&jointable.month_num<12,[5:8,10:11,17:42]).Properties.VariableNames,'1');
temptrainx= array2table(temptrainx,'VariableNames',temptrainnames);
temptrainx = temptrainx(2:end,:);
temptrainy = jointable.it(jointable.month_num>6&jointable.month_num<12);
temptrainy = temptrainy(2:end);

temptestx = lagmatrix(table2array(jointable(jointable.month_num==12,[5:8,10:11,17:42])),1);
temptestx= array2table(temptestx,'VariableNames',temptrainnames);
temptestx = temptestx(2:end,:);
temptesty = jointable.it(jointable.month_num==12);
temptesty = temptesty(2:end);

%%%%training before feature selection
x = train(:,[6:10]);
y = train.power;
etree2 = fitrtree(x,y);
etoutput2 = predict(etree2, x);
etoutput2(etoutput2<0) = 1;
trainmape = mean(abs((etoutput2-y)./y))
x = test(:,[6:10]);
y = test.power;
etoutput2 = predict(etree2, x);
etoutput2(etoutput2<0) = 1;
testmape = mean(abs((etoutput2-y)./y))


%%%%%testing importance for prediction model
rng(1);
ind = crossvalind('Kfold', numel(train.power), 5);
orgmape = [];
pimp =[];
for i=1:5
    x_train= train(ind~=i,[6:42]);
    y_train = train.power(ind~=i);
    etree2 = fitrtree(x_train,y_train);
    x= train(ind==i,[6:42]);
    y= train.power(ind==i);
    etoutput2 = predict(etree2, x);
    etoutput2(etoutput2<0) = 1;
    orgmape = [orgmape;sum(abs((etoutput2-y)./y))];
    c = predictorImportance(etree2)';
    c = (c-ones([numel(c(:,1)) 1])*min(c))./(ones([numel(c(:,1)) 1])*(max(c)-min(c)));
    pimp = [pimp,c];
end
orgmape = sum(orgmape)/numel(train.power)
pimp = sum(pimp,2);
names = train(:,[6:42]).Properties.VariableNames';

pimp = [cell2table(names),array2table(pimp),array2table([6:42]')];
pimp = sortrows(pimp,'pimp','descend');

tmape=[];
trmse=[];
ind = crossvalind('Kfold', numel(train.power), 5);
for i=1:numel(pimp.pimp)
    rng(1);
    %ind = crossvalind('Kfold', numel(train.power), 5);
    mape=0;
    rmse=0;
    for k=1:5
        x_train = train((ind~=k),pimp.Var1(1:i));          
        y_train = train.power(ind~=k);
        etree2 = fitrtree(x_train,y_train);
        x = train((ind==k),pimp.Var1(1:i));          
        y = train.power(ind==k);       
        etoutput2 = predict(etree2, x);
        etoutput2(etoutput2<0) = 1;
        mape = mape + sum(abs((etoutput2-y)./y))/numel(y);
        rmse = rmse+ sqrt(mean((y-etoutput2).^2));
    end
    tmape = [tmape;mape/5];
    trmse = [trmse;rmse/5];
end


% overfitmape=[];
% for i=1:numel(pimp.pimp)
%         rng(1);
%         x = train(:,pimp.Var1(1:i));          
%         y = train.power;     
%         etree2 = fitrtree(x,y);
%         etoutput2 = predict(etree2, x);
%         etoutput2(etoutput2<0) = 1;
%         overfitmape = [overfitmape;sum(abs((etoutput2-y)./y))/numel(y)];
% end
%    
% plot(5:37,tmape(5:37),5:37,overfitmape(5:37));

n = [tmape,trmse];
n1 = (n-ones([numel(n(:,1)) 1])*min(n))./(ones([numel(n(:,1)) 1])*(max(n)-min(n)));
opt = find(n1(:,1)==0);


figure(1);
plot(1:37,n1(:,1),'Color',[255/255,145/255,104/255],'LineWidth',3);hold on;
scatter(opt,n1(opt,1),200,'black','filled');
a = gca;
a.YLim = ([-0.03 1])
a.XTick = [1:2:40]
a.FontSize = 30;
xlabel('Top 1:n Features for Room2 LP-M');
ylabel('MAPE');

figure(2)
plot(5:37,n1(5:37,1),'Color',[255/255,144/255,104/255],'LineWidth',3);hold on;
scatter(opt,n1(opt,1),200,'black','filled');
a = gca;
a.YLim = ([-0.01 0.11])
a.XTick = [1:1:40]
a.FontSize = 20;
xlabel('Zoomed in view');



%%%prediction model after variable selection

x = train(:,pimp.Var1(1:opt));          
y = train.power;
pmodel = fitrtree(x,y);
etoutput2 = predict(pmodel, x);
etoutput2(etoutput2<0) = 1;
trainmape = [sum(abs((etoutput2-y)./y))/numel(y);]
trainrmse =sqrt(mean((y-etoutput2).^2));
x = test(:,pimp.Var1(1:opt));          
y = test.power;
etoutput2 = predict(pmodel, x);
etoutput2(etoutput2<0) = 1;
testmape = [sum(abs((etoutput2-y)./y))/numel(y);]
testrmse =sqrt(mean((y-etoutput2).^2));

% x = test(:,imp.Var1(1:16));          
% y = test.power;
% x = x(273:330,:);
% y = test.power(273:330);
% x1= x;
% x1.dt(20:30) = 20;
% etoutput2 = predict(etree2, x);
% etoutput2(etoutput2<0) = 1;
% etoutput = predict(etree2, x1);
% etoutput(etoutput2<0) = 1;
% plot(1:numel(y),etoutput2,'r',1:numel(y),y,'b',1:numel(y),etoutput,'g');
% legend('predicted','real','changed');

%%%%%temperature forecasting model

x= table2array(temptrainx);
y = temptrainy;
etree2 = fitrtree(x,y);
etoutput2 = predict(etree2, x);
etoutput2(etoutput2<0) = 1;
orgmape = [sum(abs((etoutput2-y)./y))/numel(y);]
testrmse =sqrt(mean((y-etoutput2).^2));
plot(1:200,y(1:200),'r',1:200,etoutput2(1:200),'b');
legend('real','predicted');

rng(1);
ind = crossvalind('Kfold', numel(temptrainy), 5);
orgmape = [];
fimp =[];
for i=1:5
    x_train= table2array(temptrainx(ind~=i,:));
    y_train = temptrainy(ind~=i);
    etree2 = fitrtree(x_train,y_train);
    x= table2array(temptrainx(ind==i,:));
    y= temptrainy(ind==i);
    etoutput2 = predict(etree2, x);
    etoutput2(etoutput2<0) = 1;
    orgmape = [orgmape;sum(abs((etoutput2-y)./y))];
    c = predictorImportance(etree2)';
    c = (c-ones([numel(c(:,1)) 1])*min(c))./(ones([numel(c(:,1)) 1])*(max(c)-min(c)));
    fimp = [fimp,c];
end
orgmape = sum(orgmape)/numel(train.power)
fimp = sum(fimp,2);
names = temptrainx.Properties.VariableNames';

fimp = [cell2table(names),array2table(fimp),array2table([1:numel(temptrainx(1,:))]')];
fimp = sortrows(fimp,'fimp','descend');

tmape=[];
trmse=[];
ind = crossvalind('Kfold', numel(temptrainy), 5);
for i=1:numel(fimp.fimp)
    rng(1);
    mape=0;
    rmse=0;
    for k=1:5
        x = temptrainx((ind~=k),fimp.Var1(1:i));          
        y = temptrainy(ind~=k);
        etree2 = fitrtree(x,y);
        x = temptrainx((ind==k),fimp.Var1(1:i));          
        y = temptrainy(ind==k);
        etoutput2 = predict(etree2, x);
        etoutput2(etoutput2<0) = 1;
        mape = mape + sum(abs((etoutput2-y)./y))/numel(y);
        rmse = rmse+ sqrt(mean((y-etoutput2).^2));
    end
    tmape = [tmape;mape/5];
    trmse = [trmse;rmse/5];
end
    
n = [tmape,trmse];
n = (n-ones([numel(n(:,1)) 1])*min(n))./(ones([numel(n(:,1)) 1])*(max(n)-min(n)));

temp_opt = find(n(:,1)==0);

figure(1);
plot(1:32,n(:,1),'Color',[129/255,255/255,231/255],'LineWidth',3);hold on;
scatter(temp_opt,n(temp_opt,1),200,'black','filled');
a = gca;
a.YLim = ([-0.03 1])
a.XTick = [1:2:40]
a.FontSize = 30;
xlabel('Top 1:n Features for Room2 ITF-M')
ylabel('MAPE')

figure(2)
plot(5:32,n(5:32,1),'Color',[129/255,255/255,231/255],'LineWidth',3);hold on;
scatter(temp_opt,n(temp_opt,1),200,'black','filled');
a = gca;
a.XLim = ([5 35])
a.YLim = ([-0.01 0.11])
a.XTick = [1:1:35]
a.FontSize = 20;
xlabel('Zoomed in view');


%%%% final temp forecasting model

x = temptrainx(:,fimp.Var1(1:temp_opt));          
y = temptrainy;
fmodel = fitrtree(x,y);
etoutput2 = predict(fmodel, x);
etoutput2(etoutput2<0) = 1;
trainmape = [sum(abs((etoutput2-y)./y))/numel(y);]
trainrmse =sqrt(mean((y-etoutput2).^2));
plot(1:200,y(1:200),'r',1:200,etoutput2(1:200),'b');
legend('real','predicted');

x = temptestx(:,fimp.Var1(1:temp_opt));        
y = temptesty;
etoutput2 = predict(fmodel, x);
etoutput2(etoutput2<0) = 1;
testmape = [sum(abs((etoutput2-y)./y))/numel(y);]
testrmse =sqrt(mean((y-etoutput2).^2));

plot(y(1:200));hold on;
plot(etoutput2(1:200));
legend('real','predicted');


%%%% forecasting

%% prediction variables
%     'dt''p1''tsindt''mavgit30min''mavgit2hr''tlastoncounter''tonsin''mavgdt30min''vardt''ditemp30min''mavgit1hr''changedt30min''changeit30min''mvardt2hr' 
%     'mvardt30min''mavgot2hr'

%% forecasting variables

%      'mavgit30min1''power1''mavgit1hr1''ditemp30min1''dt1''changeit30min1''dton1''tsindt1''mavgit2hr1''mavgot2hr1''mavgdt2hr1''tonsin1''mvarit2hr1' 'mvarit30min1''mvarot2hr1'
% green-47,178,67 pink-255,104,139, violet,142,155,255 beige 255 197 123 blue 99,202,255  

 
output =[];
itoutput =[];
forecast= 144;
for k = [4320:forecast:8784-forecast,35136:forecast:39600-forecast]

start = k;          %find(jointable.month_num==12)(1)


    
    farray =[];
    parray = [];
    temp = [];
    realtemp = jointable.it(start+1:start+forecast);
    real= jointable.power(start+1:start+forecast);

    dtarray = jointable.dt(start+1:start+forecast);
    %dtarray = [23.2,22,22,22,24,24,24,25,25,25,25,25]';

    otarray = jointable.ot(start+1:start+forecast);
    statusarray = jointable.power_status(start+1:start+forecast);
    vardtarray= jointable.vardt(start+1:start+forecast);

    %% initialize prediction/forecasting arrays
    startarray = jointable(start,:);
    predictarray = jointable(start,pimp.Var1(1:opt));
    forecastarray =  temptestx(1,fimp.Var1(1:temp_opt));
    itblock = jointable.it(start-12+1:start);
    otblock = jointable.ot(start-12+1:start);
    dtblock = jointable.ot(start-12+1:start);

    for i=1:forecast
        %update forcastarray for it forcasting
    % %     forecastarray.mavgit30min1= startarray.mavgit30min;
    % %     forecastarray.power1= startarray.p1;
    % %     forecastarray.mvarit1hr1= startarray.mvarit1hr;
    % %     forecastarray.ditemp30min1= startarray.ditemp30min;
    % %     forecastarray.dton1= startarray.dton;
    % %     forecastarray.dt1= startarray.dt;
    % %     forecastarray.changeit30min1= startarray.changeit30min;
    % %     forecastarray.tsindt1= startarray.tsindt;
    % %     forecastarray.mavgit2hr1= startarray.mavgit2hr;
    % %     forecastarray.tonsin1= startarray.tonsin;
    % %     forecastarray.mvarot2hr1= startarray.mvarot2hr;
    % %     forecastarray.mavgdt2hr1= startarray.mavgdt2hr;
    % %     forecastarray.mvarit2hr1= startarray.mvarit2hr;
    % %     forecastarray.mvarit30min1= startarray.mvarit30min;
    % %     forecastarray.tlastoncounter1= startarray.tlastoncounter;
    % %     forecastarray.mvarot2hr1 = startarray.mvarot2hr;
    % %     forecastarray.mavarit1hr1 = startarray.mvarit1hr;
    % %     forecastarray.changedt30min1 = startarray.changedt30min;
    % %     forecastarray.mvardt30min1 = startarray.mvardt30min;
    % %     forecastarray.mavgdt1hr1 = startarray.mavgdt1hr;
    % %     forecastarray.mavgot1hr1 = startarray.mavgot1hr;
    % %     forecastarray.tlaston1= startarray.tlaston;
    % %     forecastarray.mvardt2hr1= startarray.mvardt2hr;
    % %     forecastarray.ot1 = startarray.ot;

        tempfarray = startarray(:,[5:8,10:11,17:42]);
        forecastarray = tempfarray(:,[fimp.Var1(1:temp_opt)]);
        forecastarray.Properties.VariableNames= fimp.names(1:temp_opt)';

        farray= [farray;forecastarray];

        %%update state for power prediction
        newit = predict(fmodel, forecastarray);             %%% forecasting it
        itoutput=[itoutput;newit];
        temp = [temp;newit];
        newdt = dtarray(i);
        newot = otarray(i);
        startarray.vardt = vardtarray(i);
        newstatus = statusarray(i);
        itblock = [itblock(2:end);newit];
        otblock = [otblock(2:end);newot];
        dtblock = [dtblock(2:end);newdt];

        if startarray.power_status>0&newstatus>0
            startarray.tonsin=startarray.tonsin+1;
            startarray.tlaston=startarray.tlaston;
            startarray.tlastoncounter= startarray.tlastoncounter;
        end
        if startarray.power_status==0&newstatus>0
            startarray.tonsin=1;
            startarray.tlaston=startarray.tlastoncounter;
            startarray.tlastoncounter= startarray.tlastoncounter;
        end
        if startarray.power_status>0&newstatus==0
            startarray.tonsin=0;
            startarray.tlaston=0;
            startarray.tlastoncounter = 1;
        end
        if startarray.power_status==0&newstatus==0
            startarray.tonsin=0;
            startarray.tlaston=0;
            startarray.tlastoncounter = startarray.tlastoncounter+1;
        end

        if startarray.dt~=newdt
            startarray.tsindt=0;
        end
        if startarray.dt==newdt
            startarray.tsindt=startarray.tsindt+1;
        end

        startarray.power_status = newstatus;
        startarray.do = 2;
        startarray.dt = newdt;
        startarray.it = newit;
        startarray.dton = newdt*newstatus;
        startarray.p1 = startarray.power;
        startarray.changeot30min= mean(otblock(end-3+1:end))-startarray.mavgot30min;
        startarray.changeit30min= mean(itblock(end-3+1:end))-startarray.mavgit30min;
        startarray.changedt30min= mean(dtblock(end-3+1:end))-startarray.mavgdt30min;

        startarray.mavgot30min = mean(otblock(end-3+1:end));
        startarray.mavgot1hr = mean(otblock(end-6+1:end));
        startarray.mavgot2hr = mean(otblock(end-12+1:end));
        startarray.mvarot30min = var(otblock(end-3+1:end));
        startarray.mvarot1h = var(otblock(end-6+1:end));
        startarray.mvarot2h = var(otblock(end-12+1:end));
        startarray.mavgit30min = mean(itblock(end-3+1:end));
        startarray.mavgit1hr = mean(itblock(end-6+1:end));
        startarray.mavgit2hr = mean(itblock(end-12+1:end));
        startarray.mvarit30min = var(itblock(end-3+1:end));
        startarray.mvarit1h = var(itblock(end-6+1:end));
        startarray.mvarit2h = var(itblock(end-12+1:end));
        startarray.mavgdt30min = mean(dtblock(end-3+1:end));
        startarray.mavgdt1hr = mean(dtblock(end-6+1:end));
        startarray.mavgdt2hr = mean(dtblock(end-12+1:end));
        startarray.mvardt30min = var(dtblock(end-3+1:end));
        startarray.mvardt1h = var(dtblock(end-6+1:end));
        startarray.mvardt2h = var(dtblock(end-12+1:end));   

        startarray.ditemp30min= startarray.mavgdt30min-startarray.mvarit30min;

        %%%update predict array
    %     predictarray.dt = startarray.dt;
    %     predictarray.p1 = startarray.p1;
    %     predictarray.tsindt= startarray.tsindt;
    %     predictarray.mavgit30min = startarray.mavgit30min;
    %     predictarray.mavgit2hr = startarray.tlastoncounter;
    %     predictarray.tlastoncounter = startarray.tlastoncounter;
    %     predictarray.tonsin= startarray.tonsin;
    %     predictarray.mavgdt30min = startarray.mavgdt30min;
    %     predictarray.vardt = startarray.vardt;
    %     predictarray.ditemp30min = startarray.ditemp30min;
    %     predictarray.mavgit1hr = startarray.mavgit1hr;
    %     predictarray.changedt30min = startarray.changedt30min;
    %     predictarray.changeit30min = startarray.changeit30min;
    %     predictarray.mvardt2hr = startarray.mvardt2hr;
    %     predictarray.mvardt30min = startarray.mvardt30min;
    %     predictarray.mavgot2hr = startarray.mavgot2hr;
    
        

        predictarray=startarray(:,pimp.Var1(1:opt));
        startarray.power = predict(pmodel, predictarray);
        if newstatus==0
            startarray.power=7;
        end
        parray= [parray;predictarray];
        output = [output;startarray.power];
    end
end

y = jointable.power([4321:8784,35137:end]);
y(y==0)=7;

forecastmape = [sum(abs((output-y)./y))/numel(y);]


% numb= jointable.date_str([4321:8784,35137:end]);
% 
% find(ismember(numb,{'24-Dec-2015 00:04:45'}))
% 38449

%%%30 min ahead
% figure(3);
% subplot(2,1,1);
% plot(1:numel(y),y,'Color',[47,178,67]/255); hold on;
% plot(1:numel(y),output,'Color',[255,197,123]/255);
% legend('real','forecast','Orientation','horizontal');
% a=gca;
% a.XTick=[0,4463,8928];
% a.XTickLabel={'1 May 2015','31 May 2015 | 1 Dec 2015','31 Dec 2015'}
% ylabel('AC Power (W)')
% a.FontSize=30;
% title(strcat('30 min Forecast MAPE: ',num2str(forecastmape)));
% 
% figure(4);
% mape = abs((output-y)./y);
% mape(mape>2)=2;
% histogram(mape,100,'FaceColor',[255,197,123]/255,'EdgeColor',[255,197,123]/255,'FaceAlpha',1);
% a=gca;
% legend('APE Histogram');
% a.YScale='log';
% ylabel('Counts');
% xlabel('Absolute Percentage Error (APE)')
% a.FontSize=30;
% 
% figure(5);
% subplot(2,1,1);
% plot(7777:7920,y(7777:7920),'Color',[47,178,67]/255,'LineWidth',3); hold on;
% plot(7777:7920,output(7777:7920),'Color',[255,197,123]/255,'LineWidth',3);
% legend('real','forecast','Orientation','horizontal');
% a=gca;
% a.XTick=[7777,7920];
% a.XTickLabel={'24 Dec 2015 00:05','24 Dec 2015 23:55'}
% xlim([7770 7927])
% ylabel('AC Power (W)')
% a.FontSize=30;
% subplot(2,1,2);
% plot(7777:7920,jointable.dt(38449:38449+144-1),'g','LineWidth',3);hold on;
% plot(7777:7920,jointable.ot(38449:38449+144-1),'b','LineWidth',3);hold on;
% plot(7777:7920,jointable.it(38449:38449+144-1),'y','LineWidth',3);hold on;
% plot(7777:7920,itoutput(7777:7920),'r','LineWidth',3)
% legend('DT', 'OT', 'IT', 'Forecasted IT','Orientation','horizontal')
% a=gca;
% a.XTick=[7777,7920];
% a.XTickLabel={'24 Dec 2015 00:05','24 Dec 2015 23:55'}
% xlim([7770 7927])
% ylabel('Temperature (C)')
% a.FontSize=30;


%%%%day ahead

figure(3);
subplot(2,1,1);
plot(1:numel(y),y,'Color',[47,178,67]/255); hold on;
plot(1:numel(y),output,'Color',[99,202,255]/255);
legend('real','forecast','Orientation','horizontal');
a=gca;
a.XTick=[0,4463,8928];
a.XTickLabel={'1 May 2015','31 May 2015 | 1 Dec 2015','31 Dec 2015'}
ylabel('AC Power (W)')
a.FontSize=30;
title(strcat('1 Day Forecast MAPE: ',num2str(forecastmape)));

figure(4);
mape = abs((output-y)./y);
mape(mape>2)=2;
histogram(mape,100,'FaceColor',[99,202,255]/255,'EdgeColor',[99,202,255]/255,'FaceAlpha',1);
a=gca;
legend('APE Histogram');
a.YScale='log';
ylabel('Counts');
xlabel('Absolute Percentage Error (APE)')
a.FontSize=30;

figure(5);
subplot(2,1,1);
plot(7777:7920,y(7777:7920),'Color',[47,178,67]/255,'LineWidth',3); hold on;
plot(7777:7920,output(7777:7920),'Color',[99,202,255]/255,'LineWidth',3);
legend('real','forecast','Orientation','horizontal');
a=gca;
a.XTick=[7777,7920];
a.XTickLabel={'24 Dec 2015 00:05','24 Dec 2015 23:55'}
xlim([7770 7927])
ylabel('AC Power (W)')
a.FontSize=30;
subplot(2,1,2);
plot(7777:7920,jointable.dt(38449:38449+144-1),'g','LineWidth',3);hold on;
plot(7777:7920,jointable.ot(38449:38449+144-1),'b','LineWidth',3);hold on;
plot(7777:7920,jointable.it(38449:38449+144-1),'y','LineWidth',3);hold on;
plot(7777:7920,itoutput(7777:7920),'r','LineWidth',3)
legend('DT-1', 'OT', 'IT', 'Forecasted IT','Orientation','horizontal')
a=gca;
a.XTick=[7777,7920];
a.XTickLabel={'24 Dec 2015 00:05','24 Dec 2015 23:55'}
xlim([7770 7927])
ylabel('Temperature (C)')
a.FontSize=30;


% output_1 = output(7777:7920);  %129,255,231
% outputn = y(7777:7920);  %255,145,104
% output_11 = output(7777:7920); %178,178,178


figure(6);
subplot(2,1,1)
plot(7777:7920,jointable.dt(38449:38449+144-1),'Color',[255,145,104]/255,'LineWidth',3);hold on;
plot(7777:7920,jointable.ot(38449:38449+144-1),'b','LineWidth',3);hold on;
plot(7777:7920,jointable.it(38449:38449+144-1),'y','LineWidth',3);hold on;
plot(7777:7920,itoutput(7777:7920),'r','LineWidth',3)
legend('DT', 'OT', 'IT', 'Forecasted IT','Orientation','horizontal')
a=gca;
a.YTick=[18:2:34];
a.XTick=[7777,7920];
a.XTickLabel={'24 Dec 2015 00:05','24 Dec 2015 23:55'}
xlim([7770 7927])
ylim([18 34])
ylabel('Temperature (C)')
a.FontSize=30;

figure(7)
plot(7777:7920,output_1,'Color',[129,255,231]/255,'LineWidth',3);hold on;
plot(7777:7920,outputn,'Color',[255,145,104]/255,'LineWidth',3);hold on;
plot(7777:7920,output_11,'Color',[178,178,178]/255,'LineWidth',3);hold on;
legend('DT-1', 'DT', 'DT+1','Orientation','horizontal')
a=gca;
a.XTick=[7777,7920];
a.XTickLabel={'24 Dec 2015 00:05','24 Dec 2015 23:55'}
xlim([7770 7927])
ylabel('Power (W)')
a.FontSize=30;


sum(output_1)

