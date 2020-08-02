clear
clc
close all
%%
%load('rnn3.mat','netall3');
load('rnntanh1old.mat','netall1');
load('rnntanh4old.mat','gsremgecgtrain4');
load('rnntanh4old.mat','gsremgecgtest4');
%load('rnntanh4old.mat','seed');
 load('rnntanh4old.mat','datall');
 load('rnntanh4old.mat','class');
%load('rnntanh4old.mat','train_datall');
%load('rnntanh4old.mat','tran_dataclassall');
%%
g=gpuDevice(2);
reset(g);
 load('Hd.mat')
load('ecghd.mat')
%dpath = ['D:\Run_Wang_Document\071309_w_21'];
dpath = ['D:\Run_Wang_Document\biosignals_filtered 2\biosignals_filtered'];
dpath=string(dpath);
datall=cell(100,87);
class=cell(100,87);
gsrfeature=cell(100,87);
ecgfeature=cell(100,87);
emgfeature=cell(100,87);
% class={};
% newd=[];

dirs=dir(dpath);
dirr=struct2cell(dirs);
parfor t=1:87
    t
dirin=strcat(dpath,'\',string(dirr(1,t+2)));
dirindi=dir(dirin);
dirindi=struct2table(dirindi);
datatmp={};
classtmp={};


gsrfeattmp=[];
gsr={};
ecg={};
emg={};
ecgfeattmp=[];
emgfeattmp=[];
for i=1:1:height(dirindi)
    diri=table2array(dirindi(i,1));
    if string(diri)=='.'
       continue
    end
    if string(diri)=='..'
        continue
    end
    if string(diri)=='.DS_Store'
    continue
    end
    class2=cell2mat(diri);
    
    classtmp{i-(height(dirindi)-100),1}=[class2(13),class2(15)];

    dirreal=strcat(dirin,'\',string(diri));
    d=(readmatrix(dirreal))';
    tmp=filter(Hd,d(2,:));
     newd(1,:)=tmp(118:end);
   tmp=filter(ecghd,d(3,:));
  newd(2,:)=tmp(118:end);
   newd(3,:)=d(4,118:end);
   [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(newd(2,:),512,0);
   % [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(newd(2,:),512,1);
    ecgfeattmp(1,1)=mean(abs(diff(qrs_i_raw)));
    ecgfeattmp(1,2)=rms(abs(diff(qrs_i_raw)));
    ecgfeattmp(1,3)=std(diff(qrs_i_raw));
    emgfeattmp(1,1)=mean(newd(1,:));
    emgfeattmp(1,2)=std(newd(1,:));
    emgfeattmp(1,3)=mean(abs(diff(newd(1,:))));
    emgfeattmp(1,4)=mean(abs(diff(diff(newd(1,:)))));
 gsrfeattmp(1,1)=max(newd(1,:));
    gsrfeattmp(1,2)=mean(newd(1,:));
    gsrfeattmp(1,3)=std(newd(1,:));
    gsrfeattmp(1,4)=range(newd(1,:));
    gsrfeattmp(1,5)=iqr(newd(1,:));
    gsrfeattmp(1,6)=rms(newd(1,:));
    gsrfeattmp(1,7)=mean(abs(diff(newd(1,:))));
    gsrfeattmp(1,8)=mean(abs(diff(diff(newd(1,:)))));
     gsrfeattmp(1,11)=skewness(newd(1,:)');
     gsrfeattmp(1,12)=kurtosis(newd(1,:)');
    newd=(newd-mean(newd,2))./(std(newd')');
    
    emgfeattmp(1,5)=mean(abs(diff(newd(1,:))));
    emgfeattmp(1,6)=mean(abs(diff(diff(newd(1,:)))));
   gsrfeattmp(1,9)=mean(abs(diff(newd(1,:))));
   gsrfeattmp(1,10)=mean(abs(diff(diff(newd(1,:)))));

  
  datatmp{i-(height(dirindi)-100),1}=newd;
   
    gsr{i-(height(dirindi)-100),1}=gsrfeattmp;
    ecg{i-(height(dirindi)-100),1}=ecgfeattmp;
    emg{i-(height(dirindi)-100),1}=emgfeattmp;
    
end
    datall(:,t)=datatmp;
    class(:,t)=classtmp;
    for a=1:100
    gsrfeature(a,t)=gsr(a,1);
    ecgfeature(a,t)=ecg(a,1);
    emgfeature(a,t)=emg(a,1);
    end
end
%%
inputSize = 7;
numHiddenUnits = 100;
numClasses = 2;
trainfeat=cell(3440,1);
testfeat=cell(40,1);
sum=0;
prei=zeros(87,1);
prei2=zeros(87,1);
for k=1:87
          train2=(1:87);
        train2(k)=[];
        train_data=datall(:,train2);
        gsrtrain=gsrfeature(:,train2);
        ecgtrain=ecgfeature(:,train2);
        emgtrain=emgfeature(:,train2);
        train_dataclass=class(:,train2);
        test_data=datall(:,k);
        gsrtest=gsrfeature(:,k);
        ecgtest=ecgfeature(:,k);
        emgtest=emgfeature(:,k);
        test_dataclass=class(:,k);
    
        train_data=reshape(train_data,8600,1);
    
        gsrtrain=reshape(gsrtrain,8600,1);
       
        ecgtrain=reshape(ecgtrain,8600,1);
   
        emgtrain=reshape(emgtrain,8600,1);
     
        train_dataclass=reshape(train_dataclass,8600,1);
      
        sel=find(ismember(train_dataclass,'B1')|ismember(train_dataclass,'P1'));
        sel2=find(ismember(test_dataclass,'B1')|ismember(test_dataclass,'P1'));
        test_data=test_data(sel2);
        gsrtest=gsrtest(sel2);
        ecgtest=ecgtest(sel2);
        emgtest=emgtest(sel2);
         train_data=train_data(sel);
         gsrtrain=gsrtrain(sel);
         ecgtrain=ecgtrain(sel);
         emgtrain=emgtrain(sel);
         test_dataclass=categorical(test_dataclass(sel2));
         train_dataclass=categorical(train_dataclass(sel));
     
        trainfeature=activations(netall1{1,k},train_data,2,'MiniBatchSize',40);
        testfeature=activations(netall1{1,k},test_data,2,'MiniBatchSize',40);
      for t=1:3440
       if (ismember(train_dataclass(t,1),'B1'))
           trainfeature3(t,1)=0;
       else
           trainfeature3(t,1)=1;
       end
      end
    trainfeaturee=[];
    testfeaturee=[];
      for p=1:3440
     
      trainfeaturee=[trainfeaturee;trainfeature(:,p)', (gsrtrain{p,1}), (ecgtrain{p,1}), (emgtrain{p,1})];
      %trainfeaturee=[trainfeaturee;trainfeature(:,p)'];
      end
      for p=1:40
           testfeaturee=[testfeaturee;testfeature(:,p)',(gsrtest{p,1}), (ecgtest{p,1}), (emgtest{p,1})];
     %    testfeaturee=[testfeaturee;testfeature(:,p)'];
      end
        traintable=array2table([trainfeaturee,trainfeature3]);
        idx=fscmrmr(traintable,'Var422');
        selidx=idx(1,1:60);
        for y=1:60
        if(selidx(1,y)>400)
            sum=sum+1;
        end
        end
            for p=1:3440
             trainfeature2=trainfeaturee(p,selidx);
             trainfeat{p,1}=trainfeature2;
            
         end
         for p=1:40
            testfeature2=testfeaturee(p,selidx);
            testfeat{p,1}=testfeature2;
         end
%           layers = [
%             imageInputLayer([7 1 1])
%             fullyConnectedLayer(2,"Name","fc")
%             tanhLayer("Name","tanh")
%             softmaxLayer("Name","softmax")
%             classificationLayer("Name","classoutput")];
%             maxEpochs = 5;
%           options = trainingOptions('adam', ...
%             'ExecutionEnvironment','gpu', ...
%             'GradientThreshold',1, ...
%             'MaxEpochs',maxEpochs, ...
%              'MiniBatchSize',80,...
%             'Verbose',0, ...
%             'ValidationData',{test_data,test_dataclass}, ...
%             'ValidationFrequency',1, ...
%             'ValidationPatience',Inf,...
%             'Shuffle','every-epoch',...
%             'InitialLearnRate',0.001,...
%             'LearnRateSchedule','piecewise' );
 net=patternnet(100);
   
    for p=1:3440
        if(train_dataclass(p,1)=='B1')
            trainclassnn(1,p)=1;
            trainclassnn(2,p)=0;
        else 
            trainclassnn(1,p)=0;
            trainclassnn(2,p)=1;
        end
          for q=1:60
            trainnn(q,p)=trainfeat{p,1}(1,q);
            
          end
    end
     for p=1:40
        if(test_dataclass(p,1)=='B1')
            testclassnn(1,p)=1;
            testclassnn(2,p)=0;
        else 
            testclassnn(1,p)=0;
            testclassnn(2,p)=1;
        end
          for q=1:60
            testnn(q,p)=testfeat{p,1}(1,q);
            
          end
    end
    trainnn=normalize(trainnn,2);
    testnn=normalize(testnn,2);
    
     [net,tr]=train(net,trainnn,trainclassnn);
%      svm=fitcecoc(trainnn',trainclassnn(1,:));
%      knn=fitcknn(trainnn',trainclassnn(1,:));
     testpre=net(testnn);
%        testpre=predict(svm,testnn');
%        testpre2=predict(knn,testnn');
      for i=1:40
            if (testpre(1,i)>testpre(2,i))
                testprei(1,i)=1;
                testprei(2,i)=0;
            else
                testprei(1,i)=0;
                testprei(2,i)=1;
            end
      end
  
      [c,cm]=confusion(testclassnn,testprei);
%          net = trainNetwork(traintable,train_dataclass,layers);
           
            prei(k)=1-c;
            k
%             for c=1:40
%             prei(k,1)=prei(k,1)+(testpre(c,1)==testclassnn(1,c))./numel(test_dataclass);
%             prei2(k,1)=prei2(k,1)+(testpre2(c,1)==testclassnn(1,c))./numel(test_dataclass);
%             end
end