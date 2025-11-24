#ifndef HH_H
#define HH_H

// パラメータを構造体で定義
typedef struct {
    double E_REST;
    double C;
    double G_LEAK;
    double E_LEAK;
    double G_NA;
    double E_NA;
    double G_K;
    double E_K;
    double DT;
} HH_params;

typedef struct {
    HH_params hh;
    double G_12;
    double G_23;
} ThreeComp_params;


#define N_VARS_HH (4)
#define N_VARS_THREE_COMP (6)

static inline double alpha_m(const double v, HH_params* p) { return (2.5 - 0.1 * (v - p->E_REST)) / (exp(2.5 - 0.1 * (v - p->E_REST)) - 1.0); }
static inline double beta_m(const double v, HH_params* p) { return 4.0 * exp(-(v - p->E_REST) / 18.0); }
static inline double alpha_h(const double v, HH_params* p) { return 0.07 * exp(-(v - p->E_REST) / 20.0); }
static inline double beta_h(const double v, HH_params* p) { return 1.0 / (exp(3.0 - 0.1 * (v - p->E_REST)) + 1.0); }
static inline double alpha_n(const double v, HH_params* p) { return (0.1 - 0.01 * (v - p->E_REST)) / (exp(1 - 0.1 * (v - p->E_REST)) - 1.0); }
static inline double beta_n(const double v, HH_params* p) { return 0.125 * exp(-(v - p->E_REST) / 80.0); }

static inline double m0(const double v, HH_params* p) { return alpha_m(v, p) / (alpha_m(v, p) + beta_m(v, p)); }
static inline double h0(const double v, HH_params* p) { return alpha_h(v, p) / (alpha_h(v, p) + beta_h(v, p)); }
static inline double n0(const double v, HH_params* p) { return alpha_n(v, p) / (alpha_n(v, p) + beta_n(v, p)); }
static inline double tau_m(const double v, HH_params* p) { return 1. / (alpha_m(v, p) + beta_m(v, p)); }
static inline double tau_h(const double v, HH_params* p) { return 1. / (alpha_h(v, p) + beta_h(v, p)); }
static inline double tau_n(const double v, HH_params* p) { return 1. / (alpha_n(v, p) + beta_n(v, p)); }

static inline double dmdt(const double v, const double m, HH_params* p) { return (1.0 / tau_m(v, p)) * (-m + m0(v, p)); }
static inline double dhdt(const double v, const double h, HH_params* p) { return (1.0 / tau_h(v, p)) * (-h + h0(v, p)); }
static inline double dndt(const double v, const double n, HH_params* p) { return (1.0 / tau_n(v, p)) * (-n + n0(v, p)); }
static inline double dvdt(const double v, const double m, const double h, const double n, const double i_ext, HH_params* p)
{
    return (-p->G_LEAK * (v - p->E_LEAK) - p->G_NA * m * m * m * h * (v - p->E_NA) - p->G_K * n * n * n * n * (v - p->E_K) + i_ext) / p->C;
}


enum { V, M, H, N };

void initialize_hh(double *var, HH_params* p);
void solve_euler_hh(double *var,double i_inj, HH_params* p);

void threecomp_initialize_unified(double var[N_VARS_THREE_COMP], ThreeComp_params* p);
void solve_euler_threecomp_unified(double var[N_VARS_THREE_COMP], double i_inj, ThreeComp_params* p);

#endif // HH_H