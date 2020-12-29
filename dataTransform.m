function  [signals, labels] = dataTransform (data)

load data;

Labels = Labels.';
labels = zeros (length(Labels));
for i = 1 : length(Labels)
    if (Labels(i) == 'N') 
        labels(i) = 0;
    else
        labels(i) = 1;
    end
end
