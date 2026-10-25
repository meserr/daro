# Engineering design problems

All problems are minimisation problems with inequality constraints $g_i(x) \le 0$.
During the search the penalised objective

```math
F(x) = f(x) + \lambda \sum_i \max\{0,\, g_i(x)\}^2
```

is used with the penalty coefficient $\lambda$ given for each problem. The value reported for each run
is the objective value of the best solution found in that run that satisfies all constraints without
any tolerance. Discrete variables are rounded before every evaluation, and the rounded vector is used
both for the objective and for the constraints.

## Welded beam design

Variables: $x = [h, l, t, b]$ (weld thickness, weld length, bar height, bar thickness).

```math
\begin{aligned}
\min\; f(x) &= 1.10471\,h^{2}l + 0.04811\,tb\,(14+l)\\
g_1 &= \tau(x) - 13600\\
g_2 &= \sigma(x) - 30000\\
g_3 &= h - b\\
g_4 &= 0.10471\,h^{2} + 0.04811\,tb\,(14+l) - 5\\
g_5 &= 0.125 - h\\
g_6 &= \delta(x) - 0.25\\
g_7 &= P - P_c(x)
\end{aligned}
```

where $P = 6000$, $L = 14$, $E = 30\cdot10^{6}$, $G = 12\cdot10^{6}$ and

```math
\begin{aligned}
\tau' &= \frac{P}{\sqrt{2}\,hl}, \qquad \tau'' = \frac{MR}{J}, \qquad M = P\left(L+\frac{l}{2}\right),\\
R &= \sqrt{\frac{l^{2}}{4}+\left(\frac{h+t}{2}\right)^{2}}, \qquad
J = 2\left\{\sqrt{2}\,hl\left[\frac{l^{2}}{12}+\left(\frac{h+t}{2}\right)^{2}\right]\right\},\\
\tau &= \sqrt{\tau'^{2} + 2\tau'\tau''\frac{l}{2R} + \tau''^{2}}, \qquad
\sigma = \frac{6PL}{bt^{2}}, \qquad \delta = \frac{4PL^{3}}{Ebt^{3}},\\
P_c &= \frac{4.013\,E\sqrt{t^{2}b^{6}/36}}{L^{2}}\left(1-\frac{t}{2L}\sqrt{\frac{E}{4G}}\right).
\end{aligned}
```

Bounds: $0.1 \le h \le 2$, $0.1 \le l \le 10$, $0.1 \le t \le 10$, $0.1 \le b \le 2$.
All variables are continuous. Penalty coefficient: $\lambda = 10^{6}$.

## Pressure vessel design

Variables: $x = [T_s, T_h, R, L]$ (shell thickness, head thickness, inner radius, length of the cylindrical section).

```math
\begin{aligned}
\min\; f(x) &= 0.6224\,T_sRL + 1.7781\,T_hR^{2} + 3.1661\,T_s^{2}L + 19.84\,T_s^{2}R\\
g_1 &= -T_s + 0.0193\,R\\
g_2 &= -T_h + 0.00954\,R\\
g_3 &= -\pi R^{2}L - \tfrac{4}{3}\pi R^{3} + 1296000\\
g_4 &= L - 240
\end{aligned}
```

Bounds: $0.0625 \le T_s, T_h \le 6.1875$, $10 \le R, L \le 200$.
$T_s$ and $T_h$ are discrete and are rounded to the nearest integer multiple of 0.0625 before
evaluation; $R$ and $L$ are continuous. Penalty coefficient: $\lambda = 10^{9}$.

## Tension/compression spring design

Variables: $x = [d, D, N]$ (wire diameter, mean coil diameter, number of active coils).

```math
\begin{aligned}
\min\; f(x) &= (N+2)\,Dd^{2}\\
g_1 &= 1 - \frac{D^{3}N}{71785\,d^{4}}\\
g_2 &= \frac{4D^{2}-dD}{12566\,(Dd^{3}-d^{4})} + \frac{1}{5108\,d^{2}} - 1\\
g_3 &= 1 - \frac{140.45\,d}{D^{2}N}\\
g_4 &= \frac{d+D}{1.5} - 1
\end{aligned}
```

Bounds: $0.05 \le d \le 2$, $0.25 \le D \le 1.3$, $2 \le N \le 15$.
All variables, including $N$, are treated as continuous, as in the common formulation of this problem.
Penalty coefficient: $\lambda = 10^{6}$.

## Speed reducer design

Variables: $x = [b, m, z, l_1, l_2, d_1, d_2]$ (face width, module of teeth, number of teeth on the
pinion, lengths of the first and second shafts between bearings, diameters of the first and second shafts).

```math
\begin{aligned}
\min\; f(x) &= 0.7854\,bm^{2}(3.3333\,z^{2} + 14.9334\,z - 43.0934) - 1.508\,b\,(d_1^{2}+d_2^{2})\\
&\quad + 7.4777\,(d_1^{3}+d_2^{3}) + 0.7854\,(l_1d_1^{2}+l_2d_2^{2})\\
g_1 &= \frac{27}{bm^{2}z} - 1, \qquad g_2 = \frac{397.5}{bm^{2}z^{2}} - 1\\
g_3 &= \frac{1.93\,l_1^{3}}{mzd_1^{4}} - 1, \qquad g_4 = \frac{1.93\,l_2^{3}}{mzd_2^{4}} - 1\\
g_5 &= \frac{\sqrt{(745\,l_1/(mz))^{2} + 16.9\cdot10^{6}}}{110\,d_1^{3}} - 1\\
g_6 &= \frac{\sqrt{(745\,l_2/(mz))^{2} + 157.5\cdot10^{6}}}{85\,d_2^{3}} - 1\\
g_7 &= \frac{bm}{40} - 1, \qquad g_8 = \frac{5m}{b} - 1, \qquad g_9 = \frac{b}{12m} - 1\\
g_{10} &= \frac{1.5\,d_1 + 1.9}{l_1} - 1, \qquad g_{11} = \frac{1.1\,d_2 + 1.9}{l_2} - 1
\end{aligned}
```

Bounds: $2.6 \le b \le 3.6$, $0.7 \le m \le 0.8$, $17 \le z \le 28$, $7.3 \le l_1, l_2 \le 8.3$,
$2.9 \le d_1 \le 3.9$, $5.0 \le d_2 \le 5.5$.
$z$ is an integer variable and is rounded to the nearest integer before evaluation; the other
variables are continuous. Penalty coefficient: $\lambda = 10^{6}$.
