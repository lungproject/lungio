close all
clear
trainmrn=load('./Alldata/xptrainmrn.mat');trainmrn=trainmrn.ptrainname;
testmrn=load('./Alldata/xptestmrn.mat');testmrn=testmrn.ptestname;
hlmpdlmrn=load('./Alldata/hlmpdlmrn.mat');hlmpdlmrn=hlmpdlmrn.pname1;
hlmIOmrn=load('./Alldata/hlmretroIOmrn.mat');hlmIOmrn=hlmIOmrn.pname2;
hlmvalmrn=load('./Alldata/hlmproIOmrn.mat');hlmvalmrn=hlmvalmrn.pname3;

CIHClin = xlsread('./Alldata/CIHClin.xlsx');
HLMPDLClin= xlsread('./Alldata/HLMPDLClin.xlsx');
HLMIOClin= xlsread('./Alldata/HLMIOClin.xlsx');

trainfeature = load('./Results/predicttrain.txt');
testfeature = load('./Results/predicttest.txt');
hlmpdlfeature = load('./Results/predicthlmpdl.txt');
hlmIOfeature = load('./Results/predicthlmIO.txt');
hlmvalfeature= load('./Results/predicthlmval.txt');

pretrain = trainfeature(:,2);
pretest = testfeature(:,2);
prehlmpdl = hlmpdlfeature(:,2);
prehlmIO= hlmIOfeature(:,2);
prehlmval = hlmvalfeature(:,2);


traindata = unique(trainmrn);
for i=1:length(traindata)
    ind =  find(trainmrn==traindata(i)); 
    temp1 = pretrain(ind,:);
    trainpp(i,:)=mean(temp1); 
    [~,ind] = ismember(traindata(i),CIHClin(:,1));
    plabeltrain(i,1) = CIHClin(ind,end);

end


testdata = unique(testmrn);
for i=1:length(testdata)
    ind =  find(testmrn==testdata(i)); 
    temp1 = pretest(ind,:);
    testpp(i,:)=mean(temp1); 
    [~,ind] = ismember(testdata(i),CIHClin(:,1));
    plabeltest(i,1) = CIHClin(ind,end);

end


hlmpdldata = unique(hlmpdlmrn);
for i=1:length(hlmpdldata)
    ind =  find(hlmpdlmrn==hlmpdldata(i)); 
    temp1 = prehlmpdl(ind,:);
    hlmpdlpp(i,:)=mean(temp1); 
    [~,ind] = ismember(hlmpdldata(i),HLMPDLClin(:,1));
    plabelhlmpdl(i,1) = HLMPDLClin(ind,end);


end

cutoff=0.54 ;
evetrain = EvaluationModel(trainpp,plabeltrain,1,cutoff);
evetest = EvaluationModel(testpp,plabeltest,1,cutoff);
evehlmpdl = EvaluationModel(hlmpdlpp,plabelhlmpdl,1,cutoff);


hlmIOdata = unique(hlmIOmrn);
for i=1:length(hlmIOdata)
    ind =  find(hlmIOmrn==hlmIOdata(i)); 
    temp1 = prehlmIO(ind,:);
    hlmIOpp(i,:)=mean(temp1); 
    [~,ind] = ismember(hlmIOdata(i),HLMIOClin(:,1));
    hlmprognosis(i,:) = HLMIOClin(ind,[15 16 18 19 20]);
   
 
end

hlmprognosis(:,[1 3])=hlmprognosis(:,[1 3])/30;
hlmvaldata = unique(hlmvalmrn);
for i=1:length(hlmvaldata)
    ind =  find(hlmvalmrn==hlmvaldata(i)); 
    temp1 = prehlmval(ind,:);
    hlmvalpp(i,:)=mean(temp1); 
    [~,ind] = ismember(hlmvaldata(i),HLMIOClin(:,1));
    hlmvalprognosis(i,:) = HLMIOClin(ind,[15 16 18 19 20]);
   
 
end
hlmvalprognosis(:,[1 3])=hlmvalprognosis(:,[1 3])/30;
cutoff=0.54 ;
[~,~,~, DCBIOauc]=perfcurve(hlmprognosis(:,end),hlmIOpp, 1);
[~,~,~, DCBvalauc]=perfcurve(hlmvalprognosis(:,end),hlmvalpp, 1);





indh1 = find(hlmIOpp>=cutoff);indh2 =find(hlmIOpp<cutoff);
X1 = [hlmprognosis(indh1,1) 1-hlmprognosis(indh1,2)];
X2 = [hlmprognosis(indh2,1) 1-hlmprognosis(indh2,2)];
figure, p1=logrank(X1,X2);

X1 = [hlmprognosis(indh1,3) 1-hlmprognosis(indh1,4)];
X2 = [hlmprognosis(indh2,3) 1-hlmprognosis(indh2,4)];
figure, p2=logrank(X1,X2);


indh1 = find(hlmvalpp>=cutoff);indh2 =find(hlmvalpp<cutoff);
X1 = [hlmvalprognosis(indh1,1) 1-hlmvalprognosis(indh1,2)];
X2 = [hlmvalprognosis(indh2,1) 1-hlmvalprognosis(indh2,2)];
figure, p1=logrank(X1,X2);

X1 = [hlmvalprognosis(indh1,3) 1-hlmvalprognosis(indh1,4)];
X2 = [hlmvalprognosis(indh2,3) 1-hlmvalprognosis(indh2,4)];
figure, p2=logrank(X1,X2);


