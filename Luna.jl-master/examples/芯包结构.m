%% 精确矢量模式求解：圆形阶跃型芯包光纤
% 功能：
% 1. 求解圆形阶跃型光纤中的精确矢量模式
% 2. 计算模式有效折射率 neff
% 3. 研究 neff 随波长和纤芯半径的变化
% 4. 绘制模式的 Ex、Ey、Ez 和归一化电场强度分布
%
% 光纤模型：
%   纤芯半径：a
%   纤芯折射率：n1
%   包层折射率：n2
%   满足 n1 > n2
%
% 场的时间和传播形式：
%   exp(i*beta*z - i*omega*t)
%
% 其中 beta 为传播常数，omega 为角频率

clear;              % 清除工作区中的变量
clc;                % 清空命令行窗口
close all;          % 关闭全部图窗


%% ====================== 1. 基本参数设置 ======================

n1 = 1.450;         % 纤芯折射率
n2 = 1.444;         % 包层折射率

a0 = 4.5e-6;        % 纤芯半径，单位：m
lambda0 = 1.55e-6;  % 工作波长，单位：m

% 方位角阶数 l
%
% l = 0：
%   求解轴对称的 TE_0m 和 TM_0m 模式
%
% l >= 1：
%   求解混合矢量模式，即 HE/EH 类型模式
%
% 本程序暂时统一将它们标记为 Hybrid 模式
l_mode = 1;

% 最多保留的特征根数量
% 每一个特征根对应一个允许传播的模式
numRoots = 4;


%% ====================== 2. 单个参数下求解模式 ======================

fprintf('正在求解精确矢量模式：\n');
fprintf('波长 lambda = %.3f um\n', lambda0*1e6);
fprintf('纤芯半径 a = %.3f um\n', a0*1e6);

% 调用精确矢量模式求解函数
%
% 输入：
% n1、n2      芯层和包层折射率
% a0          纤芯半径
% lambda0     波长
% l_mode      方位角阶数
% numRoots    求解的模式数量
%
% 输出：
% modes       保存各模式信息的结构体数组
modes = solveVectorModes(n1, n2, a0, lambda0, l_mode, numRoots);

% 在命令行中输出求得的模式
disp(' ');
disp('求得的模式：');

for i = 1:length(modes)

    fprintf('%-15s  l = %d，径向序号 = %d，neff = %.10f\n', ...
        modes(i).name, ...
        modes(i).l, ...
        modes(i).m, ...
        modes(i).neff);
end


%% ====================== 3. 绘制选定模式的矢量场 ======================

% 选择要绘制的模式编号
%
% modeIndex = 1 表示绘制有效折射率最高的模式
modeIndex = 1;

% 绘制模式的 Ex、Ey、Ez 和归一化电场强度
%
% 8e-6：
%   横向计算区域为 -8 um 到 8 um
%
% 301：
%   x 和 y 方向各使用 301 个采样点
figure('Name', '精确矢量模式场分布', 'Color', 'w');

plotVectorFieldMode( ...
    n1, ...
    n2, ...
    a0, ...
    lambda0, ...
    modes(modeIndex), ...
    8e-6, ...
    301);

% 给整幅图添加总标题
sgtitle(sprintf( ...
    '%s，n_{eff}=%.8f，\\lambda=%.2f \\mum，a=%.2f \\mum', ...
    modes(modeIndex).name, ...
    modes(modeIndex).neff, ...
    lambda0*1e6, ...
    a0*1e6));


%% ====================== 4. 扫描波长 ======================

% 设置波长扫描范围
%
% 从 0.8 um 扫描到 2.0 um，共取 80 个点
lambdaList = linspace(0.8e-6, 2.0e-6, 80);

% 用于储存不同波长下各模式的有效折射率
%
% 每一行对应一个波长
% 每一列对应一个模式
neffLambda = nan(length(lambdaList), numRoots);

% 逐个波长重新求解模式
for ii = 1:length(lambdaList)

    tempModes = solveVectorModes( ...
        n1, ...
        n2, ...
        a0, ...
        lambdaList(ii), ...
        l_mode, ...
        numRoots);

    % 保存当前波长下的有效折射率
    for jj = 1:min(numRoots, length(tempModes))
        neffLambda(ii,jj) = tempModes(jj).neff;
    end
end

% 绘制有效折射率随波长的变化
figure('Name', '有效折射率随波长变化', 'Color', 'w');

hold on;
box on;
grid on;

for jj = 1:numRoots
    plot( ...
        lambdaList*1e6, ...
        neffLambda(:,jj), ...
        'LineWidth', 1.8);
end

xlabel('波长 \lambda [\mum]');
ylabel('有效折射率 n_{eff}');

title(sprintf( ...
    '精确矢量模式随波长的变化，a = %.2f \\mum，l = %d', ...
    a0*1e6, ...
    l_mode));

legend( ...
    compose('模式 %d', 1:numRoots), ...
    'Location', 'best');


%% ====================== 5. 扫描纤芯半径 ======================

% 设置纤芯半径扫描范围
%
% 从 1 um 扫描到 10 um，共取 80 个点
aList = linspace(1.0e-6, 10e-6, 80);

% 用于储存不同纤芯半径下各模式的有效折射率
neffRadius = nan(length(aList), numRoots);

% 逐个纤芯半径重新求解模式
for ii = 1:length(aList)

    tempModes = solveVectorModes( ...
        n1, ...
        n2, ...
        aList(ii), ...
        lambda0, ...
        l_mode, ...
        numRoots);

    % 保存当前纤芯半径下的有效折射率
    for jj = 1:min(numRoots, length(tempModes))
        neffRadius(ii,jj) = tempModes(jj).neff;
    end
end

% 绘制有效折射率随纤芯半径的变化
figure('Name', '有效折射率随纤芯半径变化', 'Color', 'w');

hold on;
box on;
grid on;

for jj = 1:numRoots
    plot( ...
        aList*1e6, ...
        neffRadius(:,jj), ...
        'LineWidth', 1.8);
end

xlabel('纤芯半径 a [\mum]');
ylabel('有效折射率 n_{eff}');

title(sprintf( ...
    '精确矢量模式随纤芯半径的变化，\\lambda = %.2f \\mum，l = %d', ...
    lambda0*1e6, ...
    l_mode));

legend( ...
    compose('模式 %d', 1:numRoots), ...
    'Location', 'best');


%% ====================== 6. 绘制二维参数图 ======================

% 二维扫描：
% 横轴为波长
% 纵轴为纤芯半径
% 颜色表示第一个模式的有效折射率

lambdaMap = linspace(0.8e-6, 2.0e-6, 45);
aMap = linspace(1.5e-6, 10e-6, 45);

% 初始化二维有效折射率矩阵
neffMap = nan(length(aMap), length(lambdaMap));

% 对每组纤芯半径和波长进行求解
for ia = 1:length(aMap)

    for ilam = 1:length(lambdaMap)

        tempModes = solveVectorModes( ...
            n1, ...
            n2, ...
            aMap(ia), ...
            lambdaMap(ilam), ...
            l_mode, ...
            1);

        % 如果成功求得模式，则保存第一个模式的 neff
        if ~isempty(tempModes)
            neffMap(ia, ilam) = tempModes(1).neff;
        end
    end
end

% 绘制二维有效折射率图
figure('Name', '有效折射率二维参数图', 'Color', 'w');

imagesc( ...
    lambdaMap*1e6, ...
    aMap*1e6, ...
    neffMap);

% MATLAB 的 imagesc 默认 y 轴方向向下
% 设置为正常方向，使纤芯半径从下往上增大
set(gca, 'YDir', 'normal');

colorbar;

xlabel('波长 \lambda [\mum]');
ylabel('纤芯半径 a [\mum]');

title(sprintf( ...
    '第一个精确矢量模式的 n_{eff} 二维图，l = %d', ...
    l_mode));


%% =====================================================================
%                          局部函数区域
% =====================================================================


function modes = solveVectorModes(n1, n2, a, lambda, l, numRoots)
% solveVectorModes
%
% 功能：
% 求解圆形阶跃型光纤的精确矢量模式。
%
% 对于 l = 0：
%   分别求解 TE_0m 和 TM_0m 模式。
%
% 对于 l >= 1：
%   求解混合矢量模式。
%
% 输出 modes 为结构体数组，包括：
%   name：模式名称
%   l：方位角阶数
%   m：径向根序号
%   neff：有效折射率
%   beta：传播常数

    % 真空波数
    k0 = 2*pi/lambda;

    % 导模要求：
    %
    % k0*n2 < beta < k0*n1
    %
    % 等价于：
    %
    % n2 < neff < n1
    %
    % 乘以很小的偏移量，是为了避免刚好取到边界
    betaMin = k0*n2*(1 + 1e-10);
    betaMax = k0*n1*(1 - 1e-10);

    % 将传播常数范围转换为有效折射率范围
    neffMin = betaMin/k0;
    neffMax = betaMax/k0;

    % 初始化空的模式结构体
    modes = struct( ...
        'name', {}, ...
        'l', {}, ...
        'm', {}, ...
        'neff', {}, ...
        'beta', {});

    if l == 0

        %% 求解 TE_0m 模式

        % 定义 TE 模式特征函数
        fTE = @(neff) charTE( ...
            neff, n1, n2, a, lambda);

        % 在 neff 搜索区间内寻找特征根
        rootsTE = findRoots1D( ...
            fTE, neffMin, neffMax, numRoots);

        % 将每个 TE 根保存为一个模式
        for m = 1:length(rootsTE)

            modes(end+1).name = sprintf('TE_{0%d}', m);
            modes(end).l = 0;
            modes(end).m = m;
            modes(end).neff = rootsTE(m);
            modes(end).beta = k0*rootsTE(m);
        end


        %% 求解 TM_0m 模式

        % 定义 TM 模式特征函数
        fTM = @(neff) charTM( ...
            neff, n1, n2, a, lambda);

        % 在 neff 搜索区间内寻找特征根
        rootsTM = findRoots1D( ...
            fTM, neffMin, neffMax, numRoots);

        % 将每个 TM 根保存为一个模式
        for m = 1:length(rootsTM)

            modes(end+1).name = sprintf('TM_{0%d}', m);
            modes(end).l = 0;
            modes(end).m = m;
            modes(end).neff = rootsTM(m);
            modes(end).beta = k0*rootsTM(m);
        end

        % 按 neff 从大到小排序
        %
        % 一般来说，neff 越大，模式约束越强
        [~, idx] = sort([modes.neff], 'descend');
        modes = modes(idx);

        % 最多保留 2*numRoots 个模式
        %
        % 因为 TE 和 TM 各可能有 numRoots 个根
        if length(modes) > 2*numRoots
            modes = modes(1:2*numRoots);
        end

    else

        %% 求解 l >= 1 的混合矢量模式

        % 定义混合模式特征函数
        fHybrid = @(neff) charHybrid( ...
            neff, n1, n2, a, lambda, l);

        % 寻找混合模式的特征根
        rootsHybrid = findRoots1D( ...
            fHybrid, neffMin, neffMax, numRoots);

        % 保存各个混合模式
        for m = 1:length(rootsHybrid)

            modes(end+1).name = sprintf('Hybrid_{%d%d}', l, m);
            modes(end).l = l;
            modes(end).m = m;
            modes(end).neff = rootsHybrid(m);
            modes(end).beta = k0*rootsHybrid(m);
        end

        % 按有效折射率从大到小排序
        if ~isempty(modes)
            [~, idx] = sort([modes.neff], 'descend');
            modes = modes(idx);
        end
    end
end


function val = charHybrid(neff, n1, n2, a, lambda, l)
% charHybrid
%
% 功能：
% 计算 l >= 1 时混合矢量模式的特征方程值。
%
% 当 val = 0 时，表示当前 neff 满足模式特征方程。
%
% 特征方程来自芯包边界处以下四个切向场分量连续：
%
% Ez 连续
% Hz 连续
% Ephi 连续
% Hphi 连续

    % 真空波数
    k0 = 2*pi/lambda;

    % 传播常数 beta = k0*neff
    beta = k0*neff;

    % 纤芯中的归一化横向参数
    %
    % u = a*sqrt(k0^2*n1^2 - beta^2)
    u = a*sqrt((k0*n1)^2 - beta^2);

    % 包层中的归一化衰减参数
    %
    % w = a*sqrt(beta^2 - k0^2*n2^2)
    w = a*sqrt(beta^2 - (k0*n2)^2);

    % 若 u 或 w 不为正实数，则当前 neff 不对应束缚导模
    if ~isreal(u) || ~isreal(w) || u <= 0 || w <= 0
        val = NaN;
        return;
    end

    % 计算纤芯边界 r = a 处的 Bessel 函数
    J = besselj(l, u);

    % 计算包层边界 r = a 处的修正 Bessel 函数
    K = besselk(l, w);

    % 计算对应的函数导数
    Jp = besselj_derivative(l, u);
    Kp = besselk_derivative(l, w);

    % 定义两个中间量
    %
    % 注意：
    % 这里的 A 和 B 不是场振幅，只是特征方程中的中间变量
    A = Jp/(u*J);
    B = Kp/(w*K);

    % 混合模式特征方程左端
    lhs = (A + B) * ...
          (n1^2*A + n2^2*B);

    % 混合模式特征方程右端
    rhs = (l^2 * neff^2) * ...
          (1/u^2 + 1/w^2)^2;

    % 特征函数
    %
    % 当 val = 0 时，表示满足特征方程
    val = lhs - rhs;

    % 过滤数值极点和异常值
    %
    % 当 J_l(u) 接近零时，特征函数可能出现极点
    if ~isfinite(val) || abs(val) > 1e8
        val = NaN;
    end
end


function val = charTE(neff, n1, n2, a, lambda)
% charTE
%
% 功能：
% 计算 TE_0m 模式的特征方程值。
%
% TE 模式满足：
% Ez = 0
%
% 特征方程为：
%
% J0'(u)/(u*J0(u)) + K0'(w)/(w*K0(w)) = 0

    % 真空波数和传播常数
    k0 = 2*pi/lambda;
    beta = k0*neff;

    % 计算纤芯和包层横向参数
    u = a*sqrt((k0*n1)^2 - beta^2);
    w = a*sqrt(beta^2 - (k0*n2)^2);

    % 判断是否满足导模条件
    if ~isreal(u) || ~isreal(w) || u <= 0 || w <= 0
        val = NaN;
        return;
    end

    % 计算 Bessel 函数和修正 Bessel 函数
    J = besselj(0, u);
    K = besselk(0, w);

    % 计算对应导数
    Jp = besselj_derivative(0, u);
    Kp = besselk_derivative(0, w);

    % TE 特征方程
    val = Jp/(u*J) + Kp/(w*K);

    % 排除极点和异常值
    if ~isfinite(val) || abs(val) > 1e8
        val = NaN;
    end
end


function val = charTM(neff, n1, n2, a, lambda)
% charTM
%
% 功能：
% 计算 TM_0m 模式的特征方程值。
%
% TM 模式满足：
% Hz = 0
%
% TM 特征方程与 TE 方程的主要区别是：
% 特征方程中包含折射率平方 n1^2 和 n2^2

    % 真空波数和传播常数
    k0 = 2*pi/lambda;
    beta = k0*neff;

    % 计算归一化横向参数
    u = a*sqrt((k0*n1)^2 - beta^2);
    w = a*sqrt(beta^2 - (k0*n2)^2);

    % 判断是否为束缚导模
    if ~isreal(u) || ~isreal(w) || u <= 0 || w <= 0
        val = NaN;
        return;
    end

    % 计算边界处的 Bessel 函数
    J = besselj(0, u);
    K = besselk(0, w);

    % 计算导数
    Jp = besselj_derivative(0, u);
    Kp = besselk_derivative(0, w);

    % TM 模式特征方程
    val = n1^2*Jp/(u*J) + ...
          n2^2*Kp/(w*K);

    % 排除极点和异常值
    if ~isfinite(val) || abs(val) > 1e8
        val = NaN;
    end
end


function roots = findRoots1D(fun, xmin, xmax, maxRoots)
% findRoots1D
%
% 功能：
% 在区间 [xmin, xmax] 内寻找一维函数 fun 的零点。
%
% 求根步骤：
% 1. 对整个区间进行高密度采样
% 2. 检查相邻采样点之间是否发生符号变化
% 3. 如果发生符号变化，则使用 fzero 精确求根
%
% 输入：
% fun        待求根函数
% xmin       搜索区间下限
% xmax       搜索区间上限
% maxRoots   最多寻找的根数
%
% 输出：
% roots      求得的零点数组

    % 区间采样点数
    %
    % 数值越大，越不容易漏掉根，但计算时间也会增加
    N = 12000;

    % 在搜索区间内均匀取样
    x = linspace(xmin, xmax, N);

    % 初始化函数值数组
    y = nan(size(x));

    % 逐点计算特征函数
    for i = 1:N

        try
            y(i) = fun(x(i));

        catch
            % 如果函数计算报错，则将该点设为 NaN
            y(i) = NaN;
        end
    end

    % 初始化根数组
    roots = [];

    % 遍历相邻采样点
    for i = 1:N-1

        y1 = y(i);
        y2 = y(i+1);

        % 如果某个点不是有限数，则跳过该区间
        if ~isfinite(y1) || ~isfinite(y2)
            continue;
        end

        % 如果采样点本身就是零点
        if y1 == 0

            r = x(i);

        % 如果相邻两点函数值异号，则中间可能存在根
        elseif y1*y2 < 0

            try
                % 使用 MATLAB 的 fzero 精确求根
                r = fzero(fun, [x(i), x(i+1)]);

            catch
                % 如果 fzero 求解失败，则跳过
                continue;
            end

        else
            % 未发生符号变化，继续搜索
            continue;
        end

        % 确保根位于搜索区间内
        if r > xmin && r < xmax

            % 避免重复保存同一个根
            if isempty(roots) || all(abs(r - roots) > 1e-7)
                roots(end+1) = r; %#ok<AGROW>
            end
        end

        % 达到指定根数后停止搜索
        if length(roots) >= maxRoots
            break;
        end
    end

    % 按有效折射率从大到小排列
    roots = sort(roots, 'descend');
end


function Jp = besselj_derivative(l, x)
% besselj_derivative
%
% 功能：
% 计算第一类 Bessel 函数 J_l(x) 的导数。
%
% 使用递推公式：
%
% J_l'(x) = 0.5*[J_{l-1}(x) - J_{l+1}(x)]
%
% 对于 l = 0：
%
% J_0'(x) = -J_1(x)

    if l == 0

        Jp = -besselj(1, x);

    else

        Jp = 0.5 * ...
            (besselj(l-1, x) - besselj(l+1, x));
    end
end


function Kp = besselk_derivative(l, x)
% besselk_derivative
%
% 功能：
% 计算第二类修正 Bessel 函数 K_l(x) 的导数。
%
% 使用递推公式：
%
% K_l'(x) = -0.5*[K_{l-1}(x) + K_{l+1}(x)]
%
% 对于 l = 0：
%
% K_0'(x) = -K_1(x)

    if l == 0

        Kp = -besselk(1, x);

    else

        Kp = -0.5 * ...
            (besselk(l-1, x) + besselk(l+1, x));
    end
end


function plotVectorFieldMode( ...
    n1, n2, a, lambda, mode, windowSize, N)
% plotVectorFieldMode
%
% 功能：
% 根据已经求出的 neff 和 beta，绘制精确矢量模式场。
%
% 绘制内容：
%   Re(Ex)
%   Re(Ey)
%   Re(Ez)
%   |E|^2
%
% 输入：
% n1、n2        折射率
% a             纤芯半径
% lambda        波长
% mode          模式结构体
% windowSize    横向计算区域半宽
% N             横向采样点数

    %% 基本物理常数

    c0 = 299792458;             % 真空光速
    mu0 = 4*pi*1e-7;            % 真空磁导率
    eps0 = 1/(mu0*c0^2);        % 真空介电常数

    % 真空波数
    k0 = 2*pi/lambda;

    % 角频率
    omega = 2*pi*c0/lambda;

    % 读取当前模式的传播常数和方位角阶数
    beta = mode.beta;
    l = mode.l;


    %% 建立二维横向坐标

    x = linspace(-windowSize, windowSize, N);
    y = linspace(-windowSize, windowSize, N);

    [X, Y] = meshgrid(x, y);

    % 直角坐标转换为圆柱坐标
    R = sqrt(X.^2 + Y.^2);
    Phi = atan2(Y, X);


    %% 计算各区域的波数和横向参数

    k1 = k0*n1;                 % 纤芯中的波数
    k2 = k0*n2;                 % 包层中的波数

    % 归一化横向参数
    u = a*sqrt(k1^2 - beta^2);
    w = a*sqrt(beta^2 - k2^2);

    % 各区域横向波数平方
    %
    % 纤芯内 q1sq > 0
    % 包层内 q2sq < 0
    q1sq = k1^2 - beta^2;
    q2sq = k2^2 - beta^2;


    %% 求解纵向场系数

    % 系数分别为：
    %
    % A：纤芯内 Ez 振幅
    % B：纤芯内 Hz 振幅
    % C：包层内 Ez 振幅
    % D：包层内 Hz 振幅
    coeff = longitudinalCoefficients( ...
        n1, n2, a, lambda, mode);

    A = coeff(1);
    B = coeff(2);
    C = coeff(3);
    D = coeff(4);


    %% 初始化纵向场及其径向导数

    Ez = zeros(size(R));
    Hz = zeros(size(R));

    dEzdr = zeros(size(R));
    dHzdr = zeros(size(R));

    % 区分纤芯和包层区域
    inside = R <= a;
    outside = R > a;

    % 避免在 r = 0 处计算 1/r 时出现除零
    R_safe = R;
    R_safe(R_safe == 0) = 1e-30;


    %% 计算纤芯内纵向场

    % 纤芯内径向变量
    rho1 = u*R(inside)/a;

    % 方位角因子 exp(i*l*phi)
    ang = exp(1i*l*Phi(inside));

    % 第一类 Bessel 函数及其导数
    J = besselj(l, rho1);
    Jp = besselj_derivative(l, rho1);

    % 纵向电场和磁场
    Ez(inside) = A .* J .* ang;
    Hz(inside) = B .* J .* ang;

    % 对 r 求导
    dEzdr(inside) = A .* (u/a) .* Jp .* ang;
    dHzdr(inside) = B .* (u/a) .* Jp .* ang;


    %% 计算包层内纵向场

    % 包层内径向变量
    rho2 = w*R(outside)/a;

    % 方位角因子
    ang = exp(1i*l*Phi(outside));

    % 第二类修正 Bessel 函数
    %
    % K_l 随半径增加而衰减，适合描述包层中的束缚场
    K = besselk(l, rho2);
    Kp = besselk_derivative(l, rho2);

    % 包层中的纵向电场和磁场
    Ez(outside) = C .* K .* ang;
    Hz(outside) = D .* K .* ang;

    % 对 r 求导
    dEzdr(outside) = C .* (w/a) .* Kp .* ang;
    dHzdr(outside) = D .* (w/a) .* Kp .* ang;


    %% 计算方位角导数

    % 因为场包含 exp(i*l*phi)，所以：
    %
    % d/dphi [exp(i*l*phi)] = i*l*exp(i*l*phi)
    dEzdphi = 1i*l*Ez;
    dHzdphi = 1i*l*Hz;


    %% 设置各区域介电常数和横向波数

    % 初始化为纤芯介电常数
    eps = eps0*n1^2*ones(size(R));

    % 包层区域改为包层介电常数
    eps(outside) = eps0*n2^2;

    % 初始化为纤芯横向波数平方
    qsq = q1sq*ones(size(R));

    % 包层区域改为包层横向波数平方
    qsq(outside) = q2sq;


    %% 利用 Maxwell 方程计算横向电场

    % 圆柱坐标中的径向电场 Er
    Er = -1i./qsq .* ...
        ( ...
        beta*dEzdr + ...
        omega*mu0*(1./R_safe).*dHzdphi ...
        );

    % 圆柱坐标中的方位角电场 Ephi
    Ephi = -1i./qsq .* ...
        ( ...
        beta*(1./R_safe).*dEzdphi - ...
        omega*mu0*dHzdr ...
        );


    %% 圆柱坐标转换为直角坐标

    % Ex = Er*cos(phi) - Ephi*sin(phi)
    Ex = Er.*cos(Phi) - Ephi.*sin(Phi);

    % Ey = Er*sin(phi) + Ephi*cos(phi)
    Ey = Er.*sin(Phi) + Ephi.*cos(Phi);


    %% 计算归一化电场模平方

    % 此处的 I 是电场模平方：
    %
    % |E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2
    %
    % 它不是严格的 Poynting 功率流
    I = abs(Ex).^2 + ...
        abs(Ey).^2 + ...
        abs(Ez).^2;

    % 归一化，使最大值为 1
    I = I/max(I(:));

    % 清除可能出现的无穷大或 NaN
    I(~isfinite(I)) = 0;


    %% 绘制 Ex 实部

    subplot(2, 2, 1);

    imagesc( ...
        x*1e6, ...
        y*1e6, ...
        real(Ex));

    axis image;
    colorbar;

    xlabel('x [\mum]');
    ylabel('y [\mum]');
    title('Re(E_x)');

    hold on;

    % 绘制纤芯边界
    drawCircle(a*1e6);


    %% 绘制 Ey 实部

    subplot(2, 2, 2);

    imagesc( ...
        x*1e6, ...
        y*1e6, ...
        real(Ey));

    axis image;
    colorbar;

    xlabel('x [\mum]');
    ylabel('y [\mum]');
    title('Re(E_y)');

    hold on;
    drawCircle(a*1e6);


    %% 绘制 Ez 实部

    subplot(2, 2, 3);

    imagesc( ...
        x*1e6, ...
        y*1e6, ...
        real(Ez));

    axis image;
    colorbar;

    xlabel('x [\mum]');
    ylabel('y [\mum]');
    title('Re(E_z)');

    hold on;
    drawCircle(a*1e6);


    %% 绘制归一化电场强度

    subplot(2, 2, 4);

    imagesc( ...
        x*1e6, ...
        y*1e6, ...
        I);

    axis image;
    colorbar;

    xlabel('x [\mum]');
    ylabel('y [\mum]');
    title('|E|^2，归一化');

    hold on;
    drawCircle(a*1e6);
end


function coeff = longitudinalCoefficients( ...
    n1, n2, a, lambda, mode)
% longitudinalCoefficients
%
% 功能：
% 根据芯包边界条件求纵向场的四个相对系数。
%
% 纤芯内：
%
% Ez = A*J_l(u*r/a)*exp(i*l*phi)
% Hz = B*J_l(u*r/a)*exp(i*l*phi)
%
% 包层内：
%
% Ez = C*K_l(w*r/a)*exp(i*l*phi)
% Hz = D*K_l(w*r/a)*exp(i*l*phi)
%
% 待求系数向量：
%
% coeff = [A; B; C; D]
%
% 在边界 r = a 处要求：
%
% Ez 连续
% Hz 连续
% Ephi 连续
% Hphi 连续

    %% 基本物理常数

    c0 = 299792458;
    mu0 = 4*pi*1e-7;
    eps0 = 1/(mu0*c0^2);

    k0 = 2*pi/lambda;
    omega = 2*pi*c0/lambda;

    % 读取模式参数
    beta = mode.beta;
    l = mode.l;


    %% 各区域波数和横向参数

    k1 = k0*n1;
    k2 = k0*n2;

    u = a*sqrt(k1^2 - beta^2);
    w = a*sqrt(beta^2 - k2^2);

    q1sq = k1^2 - beta^2;
    q2sq = k2^2 - beta^2;

    eps1 = eps0*n1^2;
    eps2 = eps0*n2^2;


    %% 计算边界 r = a 处的 Bessel 函数

    J = besselj(l, u);
    K = besselk(l, w);

    Jp = besselj_derivative(l, u);
    Kp = besselk_derivative(l, w);

    % 对真实半径 r 求导
    %
    % 因为函数自变量分别为 u*r/a 和 w*r/a
    dJdr = (u/a)*Jp;
    dKdr = (w/a)*Kp;


    %% 建立边界条件矩阵

    % 齐次方程形式：
    %
    % M*[A;B;C;D] = 0
    %
    % 若存在非零模式场，则矩阵 M 必须接近奇异
    M = zeros(4, 4);


    % 第一个边界条件：Ez 连续
    %
    % A*J_l(u) = C*K_l(w)
    M(1,:) = [J, 0, -K, 0];


    % 第二个边界条件：Hz 连续
    %
    % B*J_l(u) = D*K_l(w)
    M(2,:) = [0, J, 0, -K];


    % 第三个边界条件：Ephi 连续
    %
    % Ephi 由 Ez 和 Hz 的空间导数共同决定
    M(3,1) = -1i/q1sq * ...
        (beta*(1i*l/a)*J);

    M(3,2) = -1i/q1sq * ...
        (-omega*mu0*dJdr);

    M(3,3) = +1i/q2sq * ...
        (beta*(1i*l/a)*K);

    M(3,4) = +1i/q2sq * ...
        (-omega*mu0*dKdr);


    % 第四个边界条件：Hphi 连续
    %
    % Hphi 同样由纵向电磁场导数决定
    M(4,1) = -1i/q1sq * ...
        (omega*eps1*dJdr);

    M(4,2) = -1i/q1sq * ...
        (beta*(1i*l/a)*J);

    M(4,3) = +1i/q2sq * ...
        (omega*eps2*dKdr);

    M(4,4) = +1i/q2sq * ...
        (beta*(1i*l/a)*K);


    %% 使用奇异值分解求零空间向量

    % 对边界矩阵进行奇异值分解：
    %
    % M = U*S*V'
    %
    % V 的最后一列对应最小奇异值
    % 它近似满足：
    %
    % M*coeff = 0
    [~, ~, V] = svd(M);

    coeff = V(:, end);


    %% 对系数进行归一化

    % 模式场的整体振幅是任意的
    % 因此只需要得到 A、B、C、D 的相对比例
    coeff = coeff/max(abs(coeff));
end


function drawCircle(radius_um)
% drawCircle
%
% 功能：
% 在模式场图中绘制纤芯与包层的分界线。
%
% 输入 radius_um 的单位为微米。

    % 生成一个完整圆周
    t = linspace(0, 2*pi, 400);

    % 用黑色虚线绘制纤芯边界
    plot( ...
        radius_um*cos(t), ...
        radius_um*sin(t), ...
        'k--', ...
        'LineWidth', 1.2);
end