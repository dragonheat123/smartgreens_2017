clear all;
trainx = load('trainx.mat');
trainx = trainx.x;
trainy = load('trainy.mat');
trainy= trainy.y;

testx = load('testx.mat');
testx = testx.x;
testy = load('testy.mat');
testy= testy.y;


net = fitnet(10);
net.trainParam.showWindow=0; 
net = train(net,trainx',trainy');
etoutput2 = net(testx');
etoutput2(etoutput2<0) = 1;
trainmape2 = [sum(abs((etoutput2'-testy)./testy))/numel(testy);]
figure(6);
plot(3000:4000,etoutput2(3000:4000),'b',3000:4000,testy(3000:4000),'r');
legend('predicted','real');
