function [epoci] = mutaLaStanga(epoci,n)
% citim liniile din epoci
liniiEpoci = epoci(1:n,:);
% sterg liniile
epoci(1:n,:) = [];
% adaug liniile la finalul lui epoci
epoci = [epoci;liniiEpoci];
end

