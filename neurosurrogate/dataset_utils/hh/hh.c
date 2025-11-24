#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include"hh.h"

void initialize_hh(double *var, HH_params* p)
{
    double v = p->E_REST;
    var[0] = v;
    var[1] = m0(v, p);
    var[2] = h0(v, p);
    var[3] = n0(v, p);
}

void solve_euler_hh(double *var, double i_inj, HH_params* p)
{
    double v = var[0];
    double m = var[1];
    double h = var[2];
    double n = var[3];
    var[0] += dvdt(v, m, h, n, i_inj, p) * p->DT;
    var[1] += dmdt(v, m, p) * p->DT;
    var[2] += dhdt(v, h, p) * p->DT;
    var[3] += dndt(v, n, p) * p->DT;
}


void threecomp_initialize_unified(double *var, ThreeComp_params* p) {
    // var[0]〜var[3]をinitialize関数で初期化
    initialize_hh(var, &p->hh);
    
    // var[4]とvar[5]を直接初期化
    var[4] = p->hh.E_REST; // v_pre
    var[5] = p->hh.E_REST; // v_post
}

void solve_euler_threecomp_unified(double *var, double i_inj, ThreeComp_params* p) {
    // わかりやすいように変数に代入
    double v_soma = var[0];
    double v_pre = var[4];
    double v_post = var[5];

    // 軸索間の電流を計算
    double I_pre = p->G_12 * (v_pre - v_soma);
    double I_post = p->G_23 * (v_soma - v_post);

    // var_somaに相当する部分（var[0]〜var[3]）をsolve_euler関数で更新
    solve_euler_hh(var, I_pre - I_post, &p->hh);

    // v_preとv_postを更新
    var[4] += (-p->hh.G_LEAK * (v_pre - p->hh.E_LEAK) -I_pre+i_inj ) / p->hh.C * p->hh.DT;
    var[5] += (-p->hh.G_LEAK * (v_post - p->hh.E_LEAK) +I_post ) / p->hh.C * p->hh.DT;
}