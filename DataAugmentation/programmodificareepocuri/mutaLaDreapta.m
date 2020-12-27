function [epoci] = mutaLaDreapta(epoci,n)
% citim liniile din epoci
liniiEpoci = epoci(1:231-n,:);
% stergem liniile din epoci
epoci(1:231-n,:) = [];
% adaug coloanele la inceputul lui epoci
epoci = [epoci;liniiEpoci];
end

