%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Wrapper for classifying epochs using LSTM network from epoched data %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load 'C:\Users\mihai\OneDrive - Technical University of Cluj-Napoca\Teza doctorat mama\Data\ExperimentePreliminare\temporar 2ian2021\fc5_va_D_testare.mat'
% % Convert from cell to array type if needed
% f = transpose(cell2mat(Signals));
% f_tip = zeros(1, size(Labels,1));
% for i = 1 : size(Labels, 1)
%     if Labels(i) == 'N'
%         f_tip(i) = 0;
%     else
%         f_tip(i) = 1;
%     end
% end

epoch = zeros(231,1);
res = char(zeros(size(f,2),1)); 
for i = 1 : size(f,2)
    epoch = f(:,i);
    % aici facem testul
    c_epoch = mat2cell(transpose(epoch),1);
    res(i,1) = classify(net,c_epoch);
    fprintf('epoch %.0f was %.0f and classified %c\n',i,f_tip(i),res(i,1));
end
%
tp = 0;     % true positive
fp = 0;     % false negative
tn = 0;     % true negative
fn = 0;     % false negative
for i = 1:size(f,2)
    if res(i,1) == 'N' && f_tip(i) == 0
        tn = tn + 1;
    elseif res(i,1) == 'N' && f_tip(i) == 1
        fn = fn + 1;
    elseif res(i,1) == 'A' && f_tip(i) == 1
        tp = tp + 1;
    elseif res(i,1) == 'A' && f_tip(i) == 0
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

