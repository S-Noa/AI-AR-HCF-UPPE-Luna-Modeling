% 程序: firstfiber.m
% 用途: 光纤中激光脉冲传播模拟的主程序
% 主要功能: 模拟超短激光脉冲在空心光纤中的非线性传播，包括 Kerr 效应、电离效应和等离子体效应
% 模拟流程:
% 1. 设置网格参数和物理常数
% 2. 定义输入参数（脉冲、光纤、气体等）
% 3. 创建光纤模型（PM/Marcatili 模型或 FEM 模型）
% 4. 生成初始脉冲
% 5. 运行脉冲传播模拟
% 6. 保存结果
% 
% 作者: Wonkeun Chang
% 创建日期: 2013-02-08
% 最后修改: 2013-04-24

% 清除工作区和关闭图形窗口
%clearvars -except initial powerpump
close all
clear all
clc
%%

% 网格参数设置
n=2^14; % 时间窗口中的网格点数
fmax=3.0e15; % [Hz] 最大绝对频率
wl0=1030e-9; % [m] 中心波长
bandwidth=[200e-9,5000e-9]; % [m] 感兴趣的波长范围

% 物理常数定义
epsi0=8.854187817e-12; % [F/m] 真空介电常数
mu0=4*pi*1e-7; % [N/A^2] 真空磁导率
c=1/sqrt(epsi0*mu0); % [m/s] 真空中的光速

% 创建网格
numgrid=makegrid(n,fmax,wl0,bandwidth);

% 输入参数部分开始 %

usingfem=0; % 光纤模型选择：0=PM/Marcatili模型；1=FEM模型（来自COMSOL）
frog=0; % 脉冲源选择：0=理想脉冲；1=FROG数据
frogfile='100kHzpp10.h5'; % FROG数据文件名

width=120e-15; % [s] 脉冲宽度（FWHM）
bta2L=00e-30; % 泵浦脉冲的啁啾
energy=120e-6; % [J] 泵浦能量
gastype='argon'; % 气体类型
pressure=12; % [bar] 气体压力
L=1; % [m] 光纤长度
diameter=250e-6; % [m] 光纤直径
%thickness=0.3e-6; % [m] 光纤壁厚（可选）

% 模拟名称和保存设置
prefix='capillary';% COMSOL前缀
nsave=201; % 要存储的帧数
tolerance=1e-5; % 相对误差容限
savetimer=600; % [s] 保存定时器
ionmodel='adk'; % 电离模型：'none', 'adk', TODO 'ppt', 'yi', 'kfr'

% 生成模拟名称
prefixname=[prefix,'_e_',num2str(energy),'_p_',num2str(pressure)];
%prefixname=['test'];

% 噪声和温度设置
shotnoise=true; % 散粒噪声开关：'true' 或 'false'
temperature=293; % [K] 气体温度

% 输入参数部分结束 %

% 初始化条件结构体
ic.energy=energy; % 脉冲能量
ic.shotnoise=shotnoise; % 散粒噪声设置

% 系统参数结构体
sys.name=prefixname; % 模拟名称
sys.file=[sys.name,'.h5']; % 输出文件名
sys.nsave=nsave; % 要存储的帧数
sys.tol=tolerance; % 相对误差容限
sys.timer=savetimer; % 保存定时器
sys.length=L; % 光纤长度

% 开始计时
tic;

% 创建气体模型
gas=makegas(numgrid,gastype,pressure,temperature);

% 创建光纤模型
if usingfem==0
    % 使用 PM/Marcatili 模型
    kagome=makekagome(numgrid,gas,diameter,true); % 使用 Marcatili 模型
    %kagome=makekagome_PM(numgrid,gas,diameter,thickness); % 使用 PM 模型（可选）
    % % 可选：手动计算损耗
    % wl=(numgrid.wl*1e6).^2;
    % n_d=sqrt(1+(0.6961663*wl./(wl-0.0684043^2))+(0.4079426*wl./(wl-0.1162414^2))+(0.8974796*wl./(wl-9.896161^2))); % 二氧化硅折射率
    % kagome.loss=(besselzero(0,1,1)/(2*pi))^2*numgrid.wl.^2./(diameter/2).^3.*(n_d.^2+1)./sqrt(n_d.^2-1);
    % kagome.lossdB=10*log10(exp(kagome.loss));% 损耗（dB）
    % kagome.lossdB(isnan(kagome.lossdB)) = 0;
    % kagome.lossdB(isinf(kagome.lossdB)) = 0;
    % kagome.lossdB_index=interp1(numgrid.w(numgrid.bw),kagome.lossdB(numgrid.bw),numgrid.w0,'spline');
else
    % 使用 FEM 模型
    kagome=makehcfem(numgrid,diameter,gas,prefix);
    kagome.diameter=diameter; % 光纤直径
end

% 创建电离模型
ion=makeion(gastype,ionmodel);

% 生成初始脉冲
if frog==0
    % 使用理想脉冲
    %  width=24e-15; % [s] 泵浦脉冲持续时间 FWHM
    shape='gauss'; % 脉冲形状：'sech' 或 'gauss'
    ic.shape=shape;
    ic.tfwhm=width;
    [pulse0,ic]=makepulse(numgrid,ic,kagome); % 生成脉冲
else
    % 使用 FROG 数据
    [pulse0,ic]=makepulse_frog(wl0,numgrid,ic,kagome,frogfile,bta2L); % 从 FROG 数据生成脉冲
end

% 计算波长范围索引
wlrange=[min(bandwidth),max(bandwidth)];
wlib=find(numgrid.wl<wlrange(1),1,'first'); % 下限索引
wlie=find(numgrid.wl>wlrange(2),1,'last'); % 上限索引
wl=numgrid.wl(wlie:wlib); % 波长范围

% 计算峰值功率
if frog==0
    P0=energy/(2*width/1.7627); % 理想脉冲峰值功率
else
    P0=pulse0.peakp; % FROG 脉冲峰值功率
end

% 计算非线性参数
n2=2*3*gas.chi3./(8*gas.index_coef.^2*epsi0*c); % 非线性折射率
gamma_coef=numgrid.w0.*n2/c/kagome.area; % 非线性系数
L_nl=1./gamma_coef/P0 % 非线性长度

% 计算零色散波长
[~, I1]=min(abs(kagome.beta(3,wlie:wlib))); % 找到三阶色散最小的位置
kagome.zdw=wl(I1); % 零色散波长
% % 可选：计算色散参数
% beta2=(kagome.beta_coef(3));
% beta3=kagome.beta_coef(4);
% delta3=beta3/(abs(beta2)*t0);
% ss=1/(2*pi*c/wl0)/t0;
% beta_sol=kagome.beta_coef(1)+(numgrid.w-numgrid.w0)*kagome.beta_coef(2)+gamma_coef*ic.P0/2;
% delta_beta=kagome.beta(1,:)-beta_sol;
% [m,I]=min(abs(delta_beta(wlie:wlib)));
% dw_wl=wl(I);

% % 可选：绘制初始脉冲
% env=makeenv(pulse0);
% ut=env.ut;
% 
% figure(1)
% subplot(211)
% plot(numgrid.t,abs(ut).^2,'linewidth',1.2)
% xlim([-500,500]*1e-15)
% xlabel('delay')
% %ylim([0,100])
% subplot(212)
% plot(numgrid.wl*1e9,(pulse0.uf).^2,'linewidth',1.2)
% xlim([600,1000])
% xlabel('wavelength')
% 

% 绘制光纤色散和损耗
figure(2)
subplot(211)
plot(numgrid.wl*1e9,kagome.beta(3,:),'linewidth',1.2) % 绘制二阶色散
hold on
line([200 2000],[0 0],'color','k','LineStyle','--'); % 绘制参考线
xlim([200,2000]) % 设置波长范围（nm）
xlabel('wavelength')
ylim([-1,1]*1e-27) % 设置色散范围

subplot(212)
plot(numgrid.wl*1e9,kagome.lossdB(:),'linewidth',1.2) % 绘制损耗
xlim([200,2000]) % 设置波长范围（nm）
xlabel('wavelength')
hold on
%%

% 保存初始数据
filename=[sys.name,'.mat'];
save(filename,'numgrid','ic','kagome','gas','ion','sys','pulse0');

% % 可选：修改气体压力并重新计算（备用）
%pressure=10;
%gas=makegas(numgrid,gastype,pressure,temperature);

% 运行脉冲传播模拟
pulse1=propagator(pulse0,kagome,gas,ion,numgrid,sys);

% 保存传播结果
save(filename,'pulse1','-append');

% 结束计时
toc;

%%

