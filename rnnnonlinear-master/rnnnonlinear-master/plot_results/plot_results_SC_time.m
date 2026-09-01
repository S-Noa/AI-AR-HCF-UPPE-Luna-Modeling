% Plot supercontinuum temporal evolution

clear all; close all; clc

load('../results/full_SC_time_276_120e.mat')
%load('../results/full_test_results.mat')

s = size(Y_submit,2);

isteps = steps - 1;

zvec = 1:isteps+1;
ml_targets = Y_test';
ml_pred = Y_submit';

% labels
yl = 'intensity (lin)';
el = 'error (lin)';
xl = 'time (a.u.)';

ylimit = [0 0.5]; % ylim for single-shot
elimit = [-0.02 0.02]; % ylim for single-shot error
cylimit = ylimit; % colorbar limits
celimit = elimit; % colorbar limits

% select one realization
num_vec = 1:50;

eAve = zeros(1,length(num_vec));
eAveSq = zeros(1,length(num_vec));
eRMS = zeros(1,length(num_vec));

figure
for ii = 1:length(num_vec)
    num = num_vec(ii);
    
    correct_evo = ml_targets(:,(num-1)*(isteps)+1:num*(isteps));
    predicted_evo = ml_pred(:,(num-1)*(isteps)+1:num*(isteps));
    correct_evo_p = zeros(size(correct_evo,1)+1,size(correct_evo,2)+1);
    predicted_evo_p = zeros(size(predicted_evo,1)+1,size(predicted_evo,2)+1);
    correct_evo_p(1:size(correct_evo,1),1:size(correct_evo,2)) = correct_evo;
    predicted_evo_p(1:size(predicted_evo,1),1:size(predicted_evo,2)) = predicted_evo;
    
    eAve(ii) = sum(sum(abs(predicted_evo-correct_evo)));
    eAveSq(ii) = eAve(ii)^2;
    eRMS(ii) =  sqrt(sum(sum(abs(predicted_evo-correct_evo)))^2/sum(sum(abs(correct_evo)))^2);
    
    subplot(3,3,1),plot(correct_evo(:,end))
    xlabel(xl),ylabel(yl)
    xlim([1 s]), ylim(ylimit)
    title(['correct evo: ' num2str(ii)])
    
    subplot(3,3,2),plot(predicted_evo(:,end))
    xlabel(xl),ylabel(yl)
    xlim([1 s]), ylim(ylimit)
    title('predicted')
    
    subplot(3,3,3),plot(correct_evo(:,end)-predicted_evo(:,end))
    xlabel(xl),ylabel(el)
    xlim([1 s]), ylim(elimit)
    title('error')

    subplot(3,3,[4,7]),pcolor(1:s+1,zvec,correct_evo_p'), shading flat, colorbar('northoutside'), colormap jet
    caxis(cylimit);
    xlabel(xl), ylabel('distance (a.u.)')
    xlim([1 s])

    subplot(3,3,[5,8]),pcolor(1:s+1,zvec,predicted_evo_p'), shading flat, colorbar('northoutside'), colormap jet
    caxis(cylimit);
    xlabel(xl),ylabel('distance (a.u.)')
    xlim([1 s])
    
    error_m = correct_evo_p'-predicted_evo_p';
    subplot(3,3,[6,9]),pcolor(1:s+1,zvec,error_m), shading flat, colorbar('northoutside'), colormap jet
    caxis(celimit);
    xlabel(xl), ylabel('distance (a.u.)')
    xlim([1 s])
    
    set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
    pause
end

disp(['Mean absolute error: ' num2str(mean(eAve))])
disp(['Mean squared error: ' num2str(mean(eAveSq))])
disp(['Averaged root mean squared error: ' num2str(mean(eRMS)) ', std: ' num2str(std(eRMS)) ...
    ', min: ' num2str(min(eRMS)) ', max: ' num2str(max(eRMS))])
