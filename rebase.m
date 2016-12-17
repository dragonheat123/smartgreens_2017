%%%%    %%%%%%load in data   %%%%040102, 080402, 040402
clear all;
close all;
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

train = jointable(jointable.power_status==1&jointable.month_num>6&jointable.month_num<12,:);
test = [jointable(jointable.power_status==1&jointable.month_num==12,:);jointable(jointable.power_status==1&jointable.month_num==5,:)];

temptrainx = lagmatrix(table2array(train(:,[5:8,10:15,17:42])),1);
temptrainx= array2table(temptrainx,'VariableNames',temptrainnames);
temptrainx = temptrainx(2:end,:);
temptrainy = train.it(2:end);

x= table2array(temptrainx);
y = temptrainy;
etree2 = fitrtree(x,y);
etoutput2 = predict(etree2, x);
etoutput2(etoutput2<0) = 1;
orgmape = [orgmape];
testrmse =sqrt(mean((y-etoutput2).^2));
plot(1:200,y(1:200),'r',1:200,etoutput2(1:200),'b');
legend('real','predicted');


rng(1);
ind = crossvalind('Kfold', numel(temptrainy), 5);
orgmape = [];
imp =[];
for i=1:5
    x= table2array(temptrainx);
    y = temptrainy;
    etree2 = fitrtree(x,y);
    etoutput2 = predict(etree2, x);
    etoutput2(etoutput2<0) = 1;
    orgmape = [orgmape;sum(abs((etoutput2-y)./y))];
    c = predictorImportance(etree2)';
    c = (c-ones([numel(c(:,1)) 1])*min(c))./(ones([numel(c(:,1)) 1])*(max(c)-min(c)));
    imp = [imp,c];
end
orgmape = sum(orgmape)/numel(train.power)
imp = sum(imp,2);
names = temptrainx.Properties.VariableNames';

imp = [cell2table(names),array2table(imp),array2table([1:numel(temptrainx(1,:))]')];
imp = sortrows(imp,'imp','descend');

tmape=[];
trmse=[];
for i=1:numel(imp.imp)
    rng(1);
    ind = crossvalind('Kfold', numel(temptrainy), 5);
    mape=0;
    rmse=0;
    for k=1:5
        x = temptrainx((ind==k),imp.Var1(1:i));          
        y = temptrainy(ind==k);
        etree2 = fitrtree(x,y);
        etoutput2 = predict(etree2, x);
        etoutput2(etoutput2<0) = 1;
        mape = mape + sum(abs((etoutput2-y)./y))/numel(y);
        rmse = rmse+ sqrt(mean((y-etoutput2).^2));
    end
    tmape = [tmape;mape/5];
    trmse = [trmse;rmse/5];
end
    
n = [tmape,trmse];
dif = n(:,1)-[0;n(1:end-1,1)];
yyaxis left;
plot(1:numel(tmape),n(:,1));
yyaxis right;
plot(2:numel(tmape),dif(2:numel(tmape)));


%%%% generating importance
rng(1);
ind = crossvalind('Kfold', numel(train.power), 5);
orgmape = [];
imp =[];
for i=1:5
    x= train(ind==i,[6:42]);
    y = train.power(ind==i);
    etree2 = fitrtree(x,y);
    etoutput2 = predict(etree2, x);
    etoutput2(etoutput2<0) = 1;
    orgmape = [orgmape;sum(abs((etoutput2-y)./y))];
    c = predictorImportance(etree2)';
    c = (c-ones([numel(c(:,1)) 1])*min(c))./(ones([numel(c(:,1)) 1])*(max(c)-min(c)));
    imp = [imp,c];
end
orgmape = sum(orgmape)/numel(train.power)
imp = sum(imp,2);
names = train(:,[6:42]).Properties.VariableNames';

imp = [cell2table(names),array2table(imp),array2table([6:42]')];
imp = sortrows(imp,'imp','descend');

tmape=[];
trmse=[];
for i=1:numel(imp.imp)
    rng(1);
    ind = crossvalind('Kfold', numel(train.power), 5);
    mape=0;
    rmse=0;
    for k=1:5
        x = train((ind==k),imp.Var1(1:i));          
        y = train.power(ind==k);
        etree2 = fitrtree(x,y);
        etoutput2 = predict(etree2, x);
        etoutput2(etoutput2<0) = 1;
        mape = mape + sum(abs((etoutput2-y)./y))/numel(y);
        rmse = rmse+ sqrt(mean((y-etoutput2).^2));
    end
    tmape = [tmape;mape/5];
    trmse = [trmse;rmse/5];
end
    
n = [tmape,trmse];
dif = n(:,1)-[0;n(1:end-1,1)];
yyaxis left;
plot(1:37,n(:,1));
yyaxis right;
plot(2:37,dif(2:37));

n = (n-ones([numel(n(:,1)) 1])*min(n))./(ones([numel(n(:,1)) 1])*(max(n)-min(n)));
 
    
%%%testing importance for regression tree
x = train(:,imp.Var1(1:16));          
y = train.power;
etree2 = fitrtree(x,y);
etoutput2 = predict(etree2, x);
etoutput2(etoutput2<0) = 1;
trainmape = [sum(abs((etoutput2-y)./y))/numel(y);]
trainrmse =sqrt(mean((y-etoutput2).^2));
x = test(:,imp.Var1(1:16));          
y = test.power;
etoutput2 = predict(etree2, x);
etoutput2(etoutput2<0) = 1;
testmape = [sum(abs((etoutput2-y)./y))/numel(y);]
testrmse =sqrt(mean((y-etoutput2).^2));

%%%%view(etree2,'mode','graph') % graphic description



%%%testing importance for linear regression
% x = train(:,find(imp.imp>0.003)+5);
% y = train.power;
% etree2 = fitlm(table2array(x),y,'linear');
% etoutput2 = predict(etree2, table2array(x));
% etoutput2(etoutput2<0) = 1;
% orgmape = [sum(abs((etoutput2-y)./y))/numel(y);]
% orgrmse =sqrt(mean((y-etoutput2).^2));
% % 
% x = test(:,find(imp.imp>0.003)+5); 
% y = test.power;
% etoutput2 = predict(etree2, table2array(x));
% etoutput2(etoutput2<0) = 1;
% orgmape2 = [sum(abs((etoutput2-y)./y))/numel(y);]
% orgrmse2 =sqrt(mean((y-etoutput2).^2));

%%%testing importance for ensemble trees
% rng(1);
% x = train(:,find(imp>0)+5);
% y = train.power;
% etree2 = fitensemble(x,y,'LSBoost',100,'tree','type','Regression');
% etoutput2 = predict(etree2, x);
% etoutput2(etoutput2<0) = 1;
% orgmape = [sum(abs((etoutput2-y)./y))/numel(y);]
% orgrmse =sqrt(mean((y-etoutput2).^2));
% % 
% x = test(:,find(imp>0)+5); 
% y = test.power;
% etoutput2 = predict(etree2, x);
% etoutput2(etoutput2<0) = 1;
% orgmape2 = [sum(abs((etoutput2-y)./y))/numel(y);]
% orgrmse2 =sqrt(mean((y-etoutput2).^2));

%%%optimal cluster

% trainclus = [table2array(train(:,imp.Var1(1:16)))];  
% ntrainclus = (trainclus-ones([numel(trainclus(:,1)) 1])*min(trainclus))./(ones([numel(trainclus(:,1)) 1])*(max(trainclus)-min(trainclus)));
% eva = evalclusters(ntrainclus,'kmeans','DaviesBouldin','KList',[1:10]);
% 
% c= hsv(eva.OptimalK);
% figure(1);
% for i=1:eva.OptimalK
%     scatter(find(eva.OptimalY==i),train.power(eva.OptimalY==i),5,c(i,:)); hold on;
% end
% 
% testclus = [table2array(test(:,imp.Var1(1:16)))];  
% ntestclus = (testclus-ones([numel(testclus(:,1)) 1])*min(trainclus))./(ones([numel(testclus(:,1)) 1])*(max(trainclus)-min(trainclus)));
% 


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% checking optimal clusters
% 
% mape=[];
% for j=2:15                                                    %%%%number of clus to check
%     idx = kmeans(ntrainclus,j,'Start','plus');
%     totalerr=0;
%     for i=1:j
%         x1 = find(idx==i);
%         x = train(x1,imp.Var1(1:16));
%         y = train.power(x1);
%         %etree2 = fitensemble(x,y,'LSBoost',100,'tree','type','Regression');
%         etree2 = fitrtree(table2array(x),y);
%         %etree2 = fitlm(table2array(x),y,'linear');
%         etoutput2 = predict(etree2, table2array(x));
%         %etoutput2 = predict(etree2, x);
%         etoutput2(etoutput2<0) = 1;
%         etmape2 = [sum(abs((etoutput2-y)./y))/numel(y);];
%         etrmse2 =sqrt(mean((y-etoutput2).^2));
%         totalerr= totalerr + etmape2*numel(y);
%     end
%     totalerr= totalerr/numel(train.power)
%     mape = [mape;[j,totalerr,orgmape]]; 
% end

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%training for training set
% % 
% 
% rng(1);
% numclus = 7;
% color =hsv(numclus);
% totalerr=0;
% [idx,centroid] = kmeans(ntrainclus,numclus,'Start','plus');
% figure(2);
% for i=1:numclus
%     scatter(find(idx==i),train.power(idx==i),5); hold on;
% end
% 
% for i=1:numclus
%      x1 = find(idx==i);
%      x = train(x1,imp.Var1(1:16));
%      y = train.power(x1);
%         %etree2 = fitlm(table2array(x),y,'linear');                 %LINEAR model 
%         etree2 = fitrtree(table2array(x),y);            %SVR model
%         trainedtrees{i}=etree2;
%         etoutput2 = predict(etree2, table2array(x));   %linear model
%         etoutput2(etoutput2<0) = 1;
%         etmape2 = [sum(abs((etoutput2-y)./y))/numel(y);];
%         etrmse2 =sqrt(mean((y-etoutput2).^2));
%         totalerr= totalerr + etmape2*numel(y);
%  end
%  totalerr= totalerr/numel(train.power)

% %%%%%%%%%%%%%%%%%%%%%calculate which cluster the thing is in
% tnor=[];
% for k=1:numel(centroid(:,1))
%    nor= ntestclus-ones([numel(ntestclus(:,1)),1])*centroid(k,:);
%    nor = sum(abs(nor).^2,2).^(1/2);
%    tnor =[tnor,nor];
% end
% 
% [mi,id]=min(tnor,[],2);
% 
% cluslist= cell(1,numclus);
% testtotalerr= 0; 
% stitch = [];
% for i=1:numclus
%      x1 = find(id==i);
%      x = test(x1,imp.Var1(1:16));
%      y = test.power(x1);
%         etoutput2 = predict(trainedtrees{i}, table2array(x));
%         etoutput2(etoutput2<0) = 1;
%         stitch = [stitch;[x1,etoutput2]];
%         etmape2 = [sum(abs((etoutput2-y)./y))/numel(y);];
%         etrmse2 =sqrt(mean((y-etoutput2).^2));
%         if isnan(etmape2*numel(y))==0
%             testtotalerr= testtotalerr + etmape2*numel(y);
%         end
% end
%  
%  testtotalerr= testtotalerr/numel(test.power)
%  stitch = sortrows(stitch);
%  figure(3);
%  plot(stitch(300:473,2),'r');hold on;
%  plot(test.power(300:473),'b');
%  legend('predicted', 'real');
 
%%%%plot

% figure(1);
% yyaxis left;
% plot(jointable.power(38449:38592),'r');hold on;
% ylim([0 2500]);
% yyaxis right;
% plot(1:144,jointable.dton(38449:38592),'b',...
% 1:144,jointable.it(38449:38592),'g',...
% 1:144,jointable.ot(38449:38592),'cyan',...
% 1:144,jointable.tonsin(38449:38592),'black',...
% 1:144,jointable.tsindt(38449:38592),'y');
% ylim([0 60]);
% title(jointable.date_str(38449));
% legend('power','dton','it','ot','tonsin','tsindt');
% 
% figure(2);
% yyaxis left;
% plot(jointable.power(5473:5616),'r');hold on;
% ylim([0 2500]);
% yyaxis right;
% plot(1:144,jointable.dton(5473:5616),'b',...
% 1:144,jointable.it(5473:5616),'g',...
% 1:144,jointable.ot(5473:5616),'cyan',...
% 1:144,jointable.tonsin(5473:5616),'black',...
% 1:144,jointable.tsindt(5473:5616),'y');
% ylim([0 60]);
% title(jointable.date_str(5473));
% legend('power','dton','it','ot','tonsin','tsindt');
% % 


%%%%% forecasting model


