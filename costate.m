clc;
clear all;
close all;

syms p f g h k l m l_p l_f l_g l_h l_k l_l l_m u_r u_t u_n delta T c mu J2 Re


x=[p f g h k l];
u=[u_r ;u_t ;u_n];
lam=[l_p l_f l_g l_h l_k l_l];
q = 1 + f * cos(l) + g * sin(l);
s = sqrt(1 + h^2 + k^2);

A = [0; 0; 0; 0; 0; sqrt(mu * p) * (q / p)^2];

B = (1/q) * sqrt(p / 398600) * [...
        0, 2 * p, 0;
        q * sin(l), (q + 1) * cos(l) + f, -g * (h * sin(l) - k * cos(l));
       -q * cos(l), (q + 1) * sin(l) + g,  f * (h * sin(l) - k * cos(l));
        0, 0, s * cos(l) / 2;
        0, 0, s * sin(l) / 2;
        0, 0, h * sin(l) - k * cos(l)
    ];


r = p / q;
C1 = mu * J2 * Re^2 / r^4;
C2 = h * sin(l) - k * cos(l);
C3 = (1 + h^2 + k^2)^2;
ar = -3 * C1 / 2 * (1 - 12 * C2^2 / C3);
at = -12 * C1 * C2 * (h * cos(l) + k * sin(l)) / C3;
an = -6 * C1 * C2 * (1 - h^2 - k^2) / C3;
a=[ar ;at ;an];
H=lam*(A+B*a+delta*(T/m)*B*u)-l_m*delta*T/c;


lp=-diff(H,p);
lf=-diff(H,f);
lg=-diff(H,g);
lh=-diff(H,h);
lk=-diff(H,k);
ll=-diff(H,l);
lm=-diff(H,m);

disp(lp)
