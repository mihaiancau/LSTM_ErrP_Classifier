function [epoci] = modificaRandom(a,b,nrLin,nrVal,epoci)
% numaram coloanele din epoci
[~,nc] = size(epoci);
for i = 1:nc
    r = randperm(nrLin,nrVal);
    for j = 1:nrVal
        noise = a + rand * (b - a);
        epoci(r(1,j),i) = epoci(r(1,j),i) + noise;
    end
end
end

