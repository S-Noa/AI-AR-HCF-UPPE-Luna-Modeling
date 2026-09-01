% MATLAB reader for anti-resonant fibre model exports.
% Set mat_file to a generated fiber_model_YYYYmmdd_HHMMSS.mat file.
mat_file = 'fiber_model_YYYYmmdd_HHMMSS.mat';

lambda_nm = h5read(mat_file, '/lambda_nm');
D_ps_nm_km = h5read(mat_file, '/D_ps_nm_km');
beta2_ps2_km = h5read(mat_file, '/beta2_ps2_km');
loss_dB_km = h5read(mat_file, '/loss_dB_km');
resonant_wavelengths = h5read(mat_file, '/resonant_wavelengths');

figure('Color', 'w');
subplot(2,1,1);
plot(lambda_nm, D_ps_nm_km, 'LineWidth', 1.2);
hold on;
yl = ylim;
for k = 1:numel(resonant_wavelengths)
    xline(resonant_wavelengths(k), '--', 'Color', [0.3 0.3 0.8]);
end
ylim(yl);
grid on;
xlabel('\lambda (nm)');
ylabel('D (ps/(nm km))');
title('Anti-resonant fibre dispersion');

subplot(2,1,2);
semilogy(lambda_nm, max(loss_dB_km, realmin), 'LineWidth', 1.2);
grid on;
xlabel('\lambda (nm)');
ylabel('Loss (dB/km)');
title('Confinement loss');

c = 299792458;
lambda_m = lambda_nm * 1e-9;
beta2_SI = beta2_ps2_km * 1e-27;
D_from_beta2 = -(2*pi*c ./ (lambda_m.^2)) .* beta2_SI * 1e6;

figure('Color', 'w');
plot(lambda_nm, D_ps_nm_km, 'k-', 'LineWidth', 1.2);
hold on;
plot(lambda_nm, D_from_beta2, 'r--', 'LineWidth', 1.0);
grid on;
xlabel('\lambda (nm)');
ylabel('D (ps/(nm km))');
legend('Exported D', 'Converted from \beta_2', 'Location', 'best');
title('D(\lambda) and \beta_2 conversion check');
