clc;
close all;

data = csvread('resultsSeven.csv');  % Get the data from file

d = length(data(1,:)) - 1;  % Find index to ignore the quality output
for i = 1:11
    attribute = data(:,i);  % Get the column for each attribute
    attributeMean = mean(attribute);  % Find the mean of the attribute columns
    scatter(attributeMean,0.5,'filled')  % Plot the mean point
    hold on  % Prepare to plot more points
end

title('Mean Value of Wine Attributes for Quality 7')
xlabel('Attribute Value');
legend('Fixed Acidity','Volatile Acidity','Citric Acid','Residual Sugar','Chlorides','Free Sulfur Dioxide','Total Sulfur Dioxide','Density','pH','Sulphates','Alcohol','Location','NorthEast')
axis([0.05, 0.55, 0, 1])  % Set axis regions
set(gca,'ytick',[])  % Remove yAxis values
set(gca,'yticklabel',[])
