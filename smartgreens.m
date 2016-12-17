%%%%%%%%%%load in data   %%%%040102, 080402, 040402
clear all;
close all;
j = [dir('n080402.mat')];   
bigtable = []; 
for i=1:numel(j)
    load(j(i).name);  
    joindata = joindata(joindata.monthnum>=6,:);   %starts from june to december
    iton = joindata.temperature.*joindata.power_status;                 %indoor temp on
    dton = joindata.target_temperature.*joindata.power_status;          %desired temp on
   	oton = joindata.outTempCon.*joindata.power_status;                  %outside temp on
    doon= joindata.desired_operation.*joindata.power_status;            %desired operation on
    plag = lagmatrix(joindata.power_consumption,30);                
    st1 = [0;joindata.power_status(1:end-1)];
    changeit = joindata.temperature-[0;joindata.temperature(1:end-1)];
    dt1 = [0;joindata.target_temperature(1:end-1)];
    stchange = joindata.power_status-st1;
    a=find((stchange==1));
    b=find((stchange==-1));
    tfrmlaston = lagmatrix(a,1);
    tfromlaston = a-tfrmlaston;
    tfromlaston(1)=10;
    tfromlaston(tfromlaston>2880)=2880;
    tlaston = zeros([numel(joindata.power_status) 1]);                  
    tonsin = zeros([numel(joindata.power_status) 1]);
    tlastoncounter = zeros([numel(joindata.power_status) 1]);
    mavgot= sum(lagmatrix(joindata.outTempCon,[1:60]),2)/60;  
    changemavgot = mavgot-[0;mavgot(1:end-1)];
    mavg=sum(lagmatrix(joindata.temperature,[1:60]),2)/60;              %moving average for half hour
    mvar=var(lagmatrix(joindata.temperature,[1:60])')';                 %variance between for half hour
    ditemp = mavg-dton;                                                    %difference between moving average and desired temp
    changemavg = mavg-[0;mavg(1:end-1)];
    p1 = [0;joindata.power_consumption(1:end-1)];
   
    for k=1:numel(b)
        tonsin(a(k):b(k))= [1:(b(k)-a(k)+1)]';
        if numel(a)>numel(b)
            tlastoncounter(b(k)+1:a(k+1))= [1:(a(k+1)-b(k))]';
            if k+1<numel(b)
                tlaston(a(k+1):b(k+1))=max([1:(a(k+1)-b(k))]);
            end
        end
    end     
    
    joindata = [joindata,array2table(iton),array2table(dton),array2table(oton),array2table(doon),array2table(tlaston),...
        array2table(tonsin),array2table(mavg),array2table(mvar),array2table(changemavg),array2table(changeit),array2table(ditemp),...
        array2table(mavgot),array2table(changemavgot),array2table(p1)];
    lagdata = array2table(lagmatrix(table2array(joindata(:,[3,5,6,7,16,19:29])),1));
    lagdata.Properties.VariableNames=strcat(joindata(:,[3,5,6,7,16,19:29]).Properties.VariableNames,'1');
    
    joindata = [joindata, lagdata,array2table(tlastoncounter)];
    joindata = joindata(121:end,:);
    bigtable = [bigtable;joindata];
end

bigtable.power_consumption((bigtable.power_consumption==0))=1;
rng(1);

bigtable=bigtable(bigtable.power_status==1,:);

train = bigtable(bigtable.monthnum>6&bigtable.monthnum<12,:);
test = bigtable(bigtable.monthnum==12,:);

clearvars -except train test


test.power_consumption((test.power_consumption==0))=1;
train.power_consumption((train.power_consumption==0))=1;


varlist= train(:,[27,29,31,23,24]).Properties.VariableNames

%max(table2array(train(:,[27:29])))  0.0833    2.0000   31.0000            
%min(table2array(train(:,[27:29]))) -0.0833   -2.0000   -4.0000

%max(table2array(test(:,[27:29])))  0.0500    1.0000   29.0000
%min(table2array(test(:,[27:29]))) -0.0500   -1.0000   -2.7167

trainclus = [table2array(train(:,[27,29,31,23,24]))];  
ntrainclus = (trainclus-ones([numel(trainclus(:,1)) 1])*min(trainclus))./(ones([numel(trainclus(:,1)) 1])*(max(trainclus)-min(trainclus)));

testclus = [table2array(test(:,[27,29,31,23,24]))];  
ntestclus = (testclus-ones([numel(testclus(:,1)) 1])*min(trainclus))./(ones([numel(testclus(:,1)) 1])*(max(trainclus)-min(trainclus)));

eva = evalclusters(ntrainclus,'kmeans','DaviesBouldin','KList',[1:10]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% checking optimal clusters

% x = train(:,[19:31]); 
% y = train.power_consumption;
% etree2 = fitensemble(x,y,'LSBoost',100,'tree','type','Regression');
% %etree2 = fitrtree(x,y);
% etoutput2 = predict(etree2, x);
% etoutput2(etoutput2<0) = 1;
% orgmape2 = [sum(abs((etoutput2-y)./y))/numel(y);]
% orgrmse2 =sqrt(mean((y-etoutput2).^2));
% 
% mape=[];
% for j=2:15                                                      %%%%number of clus to check
%     idx = kmeans(ntrainclus,j,'Start','plus');
%     totalerr=0;
%     for i=1:j
%         x1 = find(idx==i);
%         x = train(x1,[19:31]);
%         y = train.power_consumption(x1);
%         etree2 = fitensemble(x,y,'LSBoost',100,'tree','type','Regression');
%         %etree2 = fitrtree(x,y);
%         etoutput2 = predict(etree2, x);
%         etoutput2(etoutput2<0) = 1;
%         etmape2 = [sum(abs((etoutput2-y)./y))/numel(y);];
%         etrmse2 =sqrt(mean((y-etoutput2).^2));
%         totalerr= totalerr + etmape2*numel(y);
%     end
%     totalerr= totalerr/numel(train.power_consumption)
%     mape = [mape;[j,totalerr,orgmape2]]; 
% end
% 
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%training for training set
% 
% numclus = 2;
% color =hsv(numclus);
% 
% [idx,centroid] = kmeans(ntrainclus,numclus,'Start','plus');
% 
% 
% trainedtrees = cell(numclus,1);
% 
% cluslist= cell(1,numclus);
% varlist= train(:,[27,29,31,23,24]).Properties.VariableNames
%     for i=1:numclus
%         x1 = find(idx==i);
%         figure(1);
%         subplot(2,2,1)
%         title('Power Consumption and its clusters')
%         scatter(x1,train.power_consumption(x1),5,color(i,:),'filled'); hold on;
%         cluslist{i}=num2str(i);
%         figure(2);
%         scatter3(train.ditemp(x1),train.changemavg(x1),train.changemavgot(x1),5,color(i,:),'filled');hold on;
%         xlabel('ditemp'); ylabel('changemavg');
%         for g=1:numel(trainclus(1,:))
%             figure(1)
%             subplot(2,2,1+g);
%             scatter(trainclus(idx==i,g),train.power_consumption(idx==i),5,color(i,:),'filled'); hold on;
%         end
%     end
%     legend(cluslist);
%     title('Power Consumption')
%     for g=1:numel(trainclus(1,:))
%         figure(1);
%         subplot(2,2,1+g);
%         title(varlist(g));
%         legend(cluslist);
%         ylabel('Power Consumption');
%         xlabel(varlist(g));
%     end
% totalerr= 0;
% varlist2= train(:,[19:29]).Properties.VariableNames 
% plotnum = ceil(numclus/5);
% for i=1:numclus
%      x1 = find(idx==i);
%      x = train(x1,[19:29]);
%      y = train.power_consumption(x1);
%         etree2 = fitensemble(x,y,'LSBoost',100,'tree','type','Regression');
%         %etree2 = fitlm(table2array(x),y);                 LINEAR model 
%         trainedtrees{i}=etree2;
%         %etoutput2 = predict(etree2, table2array(x));   linear model
%         etoutput2 = predict(etree2, x); 
%         etoutput2(etoutput2<0) = 1;
%         etmape2 = [sum(abs((etoutput2-y)./y))/numel(y);];
%         etrmse2 =sqrt(mean((y-etoutput2).^2));
%         figure(2);
%         subplot(plotnum,5,i);
%         plot(1:numel(y),y,'r',1:numel(y),etoutput2,'b')
%         plot(1:numel(y),y,'black');hold on;
%         plot(1:numel(y),etoutput2,'Color',color(i,:))
%         legend('real','predicted');
%         title(strcat('c-',num2str(i),'-',num2str(etmape2),'-',num2str(numel(y))));
%         totalerr= totalerr + etmape2*numel(y);
%  end
%  totalerr= totalerr/numel(train.power_consumption)
% 
% % %%%%%%%%%%%%%%%%%%%%%calculate which cluster the thing is in
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
% varlist= test(:,[27,29]).Properties.VariableNames
%     for i=1:numclus
%         x1 = find(id==i);
%         figure(3);
%         subplot(2,2,1)
%         title('Power Consumption and its clusters test')
%         scatter(x1,test.power_consumption(x1),5,color(i,:),'filled'); hold on;
%         cluslist{i}=num2str(i);
%         subplot(2,2,4)
%         scatter(test.ditemp(x1),test.changemavg(x1),5,color(i,:),'filled');hold on;
%         xlabel('ditemp'); ylabel('changemavg');
%         for g=1:numel(testclus(1,:))
%             figure(3)
%             subplot(2,2,1+g);
%             scatter(testclus(id==i,g),test.power_consumption(id==i),5,color(i,:),'filled'); hold on;
%         end
%     end
%     legend(cluslist);
%     title('Power Consumption')
%     for g=1:numel(testclus(1,:))
%         figure(3);
%         subplot(2,2,1+g);
%         title(varlist(g));
%         legend(cluslist);
%         ylabel('Power Consumption');
%         xlabel(varlist(g));
%     end
% testtotalerr= 0;
% varlist2= test(:,[19:29]).Properties.VariableNames   
% stitch = [];
% for i=1:numclus
%      x1 = find(id==i);
%      x = test(x1,[19:29]);
%      y = test.power_consumption(x1);
%         etoutput2 = predict(trainedtrees{i}, x);
%         etoutput2(etoutput2<0) = 1;
%         stitch = [stitch;[x1,etoutput2]];
%         etmape2 = [sum(abs((etoutput2-y)./y))/numel(y);];
%         etrmse2 =sqrt(mean((y-etoutput2).^2));
%         figure(4);
%         subplot(plotnum,5,i);
%         plot(1:numel(y),y,'r',1:numel(y),etoutput2,'b')
%         plot(1:numel(y),y,'black');hold on;
%         plot(1:numel(y),etoutput2,'Color',color(i,:))
%         legend('real','predicted');
%         title(strcat('c-',num2str(i),'-',num2str(etmape2),'-',num2str(numel(y))));
%         if isnan(etmape2*numel(y))==0
%             testtotalerr= testtotalerr + etmape2*numel(y);
%         end
%  end
%  testtotalerr= testtotalerr/numel(test.power_consumption)
% 
% stitch = sortrows(stitch);
% x = train(:,[19:29]); 
% y = train.power_consumption;
% etree2 = fitensemble(x,y,'LSBoost',100,'tree','type','Regression');
% x = test(:,[19:29]); 
% y = test.power_consumption;
% etoutput2 = predict(etree2, x);
% etoutput2(etoutput2<0) = 1;
% testmape2 = [sum(abs((etoutput2-y)./y))/numel(y);]
% figure(5);
% yyaxis left;
% plot(66241:72000,etoutput2(66241:72000),'r',66241:72000,y(66241:72000),'b',66241:72000,stitch(66241:72000,2),'y');
% yyaxis right;
% plot(66241:72000,test.dton(66241:72000),'g',66241:72000,test.oton(66241:72000),'black',66241:72000,test.iton(66241:72000),'cyan');
% %plot(1:numel(y),etoutput2,'r',1:numel(y),y,'b',1:numel(y),stitch(:,2),'y')
% legend('predicted','real','clustered','dton','oton','iton');
% % 
% %%%%%%%%%%%%%%%%%%%%%%%line plot for cluster
% 
% % figure(6)
% % for i=1:numclus
% %     numlist = [1:numel(train.power_consumption)]';
% %     numlist = array2table(numlist);
% %     x1= find(idx==i);
% %     tab = [array2table(x1), array2table(train.power_consumption(x1))];
% %     tab.Properties.VariableNames{1}='numlist';
% %     c = outerjoin(numlist,tab);
% %     c= table2array(c);
% %     plot(c(:,1),c(:,3),'Color',color(i,:),'LineWidth',2);hold on;
% % end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%importance in each ensemble tree
% imp=[];
% 
% for i= 1:numclus
%     imp = [imp;predictorImportance(trainedtrees{i})];
% end
% 
% x = train(:,[19:29]); 
% y = train.power_consumption;
% etree2 = fitensemble(x,y,'LSBoost',100,'tree','type','Regression');
% etoutput2 = predict(etree2, x);
% etoutput2(etoutput2<0) = 1;
% trainmape2 = [sum(abs((etoutput2-y)./y))/numel(y);]
% 
% imp= [imp;predictorImportance(etree2)]
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%forecasting model

%%'p1'  'temperature1'    'iton1'    'dton1'    'tonsin1'    'changemavg1'    'changeit1'    'ditemp1'
% x= train(:,[30,37,35,41:46]); 
% y = train.temperature;
% tempredict = fitlm(table2array(x),y);                  %linear
% etoutput2 = round(predict(tempredict, table2array(x)));
% etoutput2(etoutput2<0) = 1;
% trainmape2 = [sum(abs((etoutput2-y)./y))/numel(y);]
% 
% x = test(:,[30,37,35,41:46]); 
% 
% y = test.temperature;
% etoutput2 = round(predict(tempredict, table2array(x)));
% etoutput2(etoutput2<0) = 1;
% trainmape2 = [sum(abs((etoutput2-y)./y))/numel(y);]
% figure(6);
% plot(3000:4000,etoutput2(3000:4000),'b',3000:4000,y(3000:4000),'r');
% legend('predicted','real');
% 
% %%%%%%%% reconstructing series
% % % 'iton'    'dton'    'oton'    'doon'    'tlaston'    'tonsin'    'mavg'    'mvar'    'changemavg'    'changeit'    'ditemp'
% % % input--> 'dton', 'doon',
% % % auto update --> 'tlaston', 'tonsin', 'oton'
% % % prediction--> 'iton', 'mavg', 'mvar', 'changemavg', 'changeit', 'ditemp'
% %%% forecasting--> 'temperature1'    'iton1'    'dton1'    'tonsin1'    'changemavg1'    'changeit1'    'ditemp1'
% 
% 
% % figure(10);
% % yyaxis left
% % plot(test.power_consumption(test.weeknum==53&test.daytype==4),'b');
% % yyaxis right
% % plot(test.dton(test.weeknum==53&test.daytype==4),'g');hold on;
% % plot(test.oton(test.weeknum==53&test.daytype==4),'black');hold on;
% % plot(test.iton(test.weeknum==53&test.daytype==4),'cyan');
% % ylim([0 35])
% % ploty = gca;
% % ploty.YTick =[0:1:35];
% % legend('power consumption', 'dton','oton','iton')
% % 
% start = 66241;          %24 dec 2015, 000hrs
% forecast= 60;
% %start = 83521;
% 
% % dtarray=[];
% % otarray = [];
% % doarray=[];
% dtarray = test.dton(start+1:start+forecast);
% otarray = test.oton(start+1:start+forecast);
% doarray = test.doon(start+1:start+forecast);
% 
% oldtimelastoncounter = test(start,[47]);
% oldfarray = test(start,[33,36,37,41,44:46]);    %%%for forecasting
% oldparray = test(start,[19:29]);                %%%for prediction
% temparray = test.temperature(start-60+1:start);
% forecastoutput = [];
% mforecast = [];
% mpredict = [];
% clust = [];
% for i=1:forecast
%     %% control inputs for t+1
%     newoton = otarray(i);
%     newdoon = doarray(i);
%     newdton = dtarray(i);
%     %%update oldfarray for mavg forcasting
%     oldfarray.temperature1 = temparray(60);
%     oldfarray.iton1 = oldparray.iton;
%     oldfarray.dton1 = oldparray.dton;
%     oldfarray.tonsin1 = oldparray.tonsin;
%     oldfarray.changemavg1 = oldparray.changemavg;
%     oldfarray.changeit1 = oldparray.changeit;
%     oldfarray.ditemp1= oldparray.ditemp;
%     
%     newmavg = predict(tempredict, table2array(oldfarray));
%     mforecast = [mforecast; [table2array(oldfarray),newmavg]];
%     %%calculate new temp related variables
%     newtemp = round((newmavg*60+temparray(1))-sum(temparray));
%     newchangeit = newtemp-temparray(60);
%     temparray = [temparray(2:end); newtemp];            
%     newmvar= var(temparray);
%     newchangemavg= newmavg-oldparray.mavg;
%     newditemp = newmavg - newdton;
%     if dtarray(i)>0
%         newiton = newtemp;
%     else
%         newiton = 0;
%     end
%     if oldparray.dton>0 & newdton>0
%         newtlaston = oldparray.tlaston;
%         newtonsin = oldparray.tonsin+1;
%         newtimelastoncounter=0;
%     end
%     if oldparray.dton==0 & newdton>0
%         newtonsin = 1;
%         newtlaston = newtimelastoncounter;
%         newtimelsatoncounter=0;
%     end
%     if oldparray.dton>0 & newdton==0
%         newtonsin = 0;
%         newtlaston = 0;
%         newtimelastonsounter=1;
%     end
%     if oldparray.dton==0 & newdton==0
%         newtimelastoncounter = oldtimelastoncounter + 1;
%         newtonsin =0;
%         newtlaston =0;
%     end
%     
%     oldtimelastoncounter = newtimelastoncounter;    %%updatetimelastoncounter
% 
%     %%%update oldparray for power prediction
%     oldparray.iton = newiton;
%     oldparray.dton = newdton;
%     oldparray.oton = newoton;
%     oldparray.doon = newdoon;
%     oldparray.tlaston = newtlaston;
%     oldparray.tonsin = newtonsin;
%     oldparray.mavg = newmavg;
%     oldparray.mvar = newmvar;
%     oldparray.changemavg = newchangemavg;
%     oldparray.ditemp = newditemp;
%     oldparray.changeit = newchangeit;
%     mpredict = [mpredict;oldparray];
% 
%     
%     pclus = [newchangemavg,newditemp];
%     npclus = (pclus-ones([numel(pclus(:,1)) 1])*min(trainclus))./(ones([numel(pclus(:,1)) 1])*(max(trainclus)-min(trainclus)));
%     tnor= [];
%     for k=1:numel(centroid(:,1))
%        nor= npclus-ones([numel(npclus(:,1)),1])*centroid(k,:);
%        nor = sum(abs(nor).^2,2).^(1/    2);
%        tnor =[tnor,nor];
%     end
%     [mi,id]=min(tnor,[],2);
%     clust= [clust;id];
%     forecastoutput = [forecastoutput; predict(trainedtrees{id}, oldparray)];     
% end
% 
% yyaxis left;
% plot(1:forecast,test.power_consumption(start+1:start+forecast),'r',1:forecast,forecastoutput,'b');
% ylim([-100 2500]);
% yyaxis right;
% plot(1:forecast,dtarray,'green',1:forecast,mpredict.iton(1:forecast),'black')
% ylim([-1 50]);
% a= gca;
% a.YTick = [-1:1:50]
% legend('real','forecasted','dt','it');
