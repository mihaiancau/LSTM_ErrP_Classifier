%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Wrapper for classifying epochs using LSTM network %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load 'RawData_Testing\s11_va_fc6.mat';
[moms,~] = size(moment);                              
res = char(zeros(60,1));                            

k = 1;      % index pentru moment(k,1)
epoch = zeros(231,1);
for t = 1 : length (fc6)
    if t == moment(k,1)
        epoch_start = moment(k,1) - 26;
        epoch_end = moment(k,1) + 205;
        for i = epoch_start+1 : epoch_end
            epoch(i-epoch_start,1) = fc6(i,1);
        end
        
        % aici facem testul
        c_epoch = mat2cell(transpose(epoch),1);
        res(k,1) = classify(net,c_epoch);
        fprintf('epoch %.0f was %.0f and classified %c\n',k,moment(k,2),res(k,1));
        
        if k < length(moment)
            k = k + 1;
        end
     end
end
%
tp = 0;     % true positive
fp = 0;     % false negative
tn = 0;     % true negative
fn = 0;     % false negative
for i = 1:60
    if res(i,1) == 'N' && moment(k,2) == 0
        tn = tn + 1;
    elseif res(i,1) == 'N' && moment(i,2) == 1
        fn = fn + 1;
    elseif res(i,1) == 'A' && moment(i,2) == 1
        tp = tp + 1;
    elseif res(i,1) == 'A' && moment(i,2) == 0
        fp = fp + 1;
    end
end
% accuracy: cate au fost prezise corect din total (este bine sa fie mare)
accuracy = (tp + tn) / (tp +fp + tn + fn);
fprintf('Accuracy: %.2f\n',accuracy);
error = (fp + fn) / (tp +fp + tn + fn);
fprintf('Error: %.2f\n',error);
% recall: dintre 'pozitive', cate au fost prezise corect (este bine sa fie mare)
sensitivity = tp / (tp + fn); % recall = sensitivity = true-positive rate
fprintf('Sensitivity = Recall: %.2f\n',sensitivity);
specificity = tn / (tn + fp);
fprintf('Specificity: %.2f\n',specificity);
% precision: dintre toate 'pozitivele' prezise corect, cate sunt de fapt pozitive
ppv = tp / (tp + fp); % precision = positive-predictive value
fprintf('Precision = Positive-predictive value: %.2f\n',ppv);
npv = tn / (tn + fn); % negative-predictive value
fprintf('Negative-predictive value: %.2f\n',npv);

