load('fc5_vizActiv_d.mat');
% functia 'adaugaLaFinal' face o translatie la stanga 
% a semnalului, cu n pozitii
% n = 4 echivaleaza cu 4*3.9 = 15.6 ms
n = 2;
signal = fc5;
typeSignal = fc5_tip;
[nrLin,nc] = size(fc5);
semnal_1 = signal;
tipSemnal_1 = typeSignal;
%  mutare semnal la stanga
for i = 1:n
    epoci = signal;
    epoci1 = mutaLaStanga(epoci,n);
    semnal_1 = [semnal_1 epoci1];
    tipSemnal_1 = [tipSemnal_1 typeSignal];
end
%
semnal_2 = signal;
tipSemnal_2 = typeSignal;
% mutare semnal la dreapta
for i = 1:n
    epoci = signal;
    epoci2 = mutaLaDreapta(epoci,n);
    semnal_2 = [semnal_2 epoci2];
    tipSemnal_2 = [tipSemnal_2 typeSignal];
end
% random noise intre (-5,5), pentru 10% dintre valorile unui epoc
% r = randperm(213,21);     % random integers without repeating
% noise = -5 + rand*10;         % random float between (-5,5)
% k = factorul de multiplicare random al epocilor
k = 2;
a = -6;
b = 6;
semnal_3 = signal;
epoci = signal;
% nrVal = numarul de valori modificate aleator (procente)
nrVal = floor(0.1*nrLin);
tipSemnal_3 = typeSignal;
for i = 1:k
    epoci3 = modificaRandom(a,b,nrLin,nrVal,epoci);
    semnal_3 = [semnal_3 epoci3];
    tipSemnal_3 = [tipSemnal_3 typeSignal];
end
% repetare identica a semnalului
semnal_4 = [signal signal];% signal signal signal];
tipSemnal_4 = [typeSignal typeSignal];% typeSignal typeSignal typeSignal];
semnal = [semnal_1 semnal_2 semnal_3 semnal_4];
tip_semnal = [tipSemnal_1 tipSemnal_2 tipSemnal_3 tipSemnal_4];
% conversia datelor in format 'cell'
[mSemnal,nSemnal] = size(semnal);
[mTip,nTip] = size(tip_semnal);
Signals = num2cell(semnal,1);
Signals = transpose(Signals);
Signals = cellfun(@transpose, Signals, 'UniformOutput', false);
%
Labels = cell(nTip,1);
for i = 1:nTip
    if tip_semnal(1,i) == 0
        Labels{i,1} = 'N';
    else
        Labels{i,1} = 'A';
    end
end
Labels = categorical(Labels);
save fc5_va_D.mat Labels Signals;


