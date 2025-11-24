
#define NC_TRAUB ( 19    ) // # compartments
#define DT ( 0.01  ) // 10 mu-sec


#define T  ( 200   ) // ms ここは使われない
#define NT ( 20000 ) // T / DT　ここは使われない

#define Cm   ( 3.0 )  // micro F / cm^2
#define Ri   ( 0.1 )  // K Ohm-cm
#define Rm   ( 10.0 ) // K Ohm-cm^2 (unused)
#define Beta ( 0.075 )

#define V_LEAK ( -60.0 )
#define V_Na ( ( 115.0 + ( V_LEAK ) ) )
#define V_Ca ( ( 140.0 + ( V_LEAK ) ) )
#define V_K  ( ( -15.0 + ( V_LEAK ) ) )

enum { V, XI, M, S, N, C, A, H, R, B, Q, N_VARS_TRAUB };

static inline double pir2 ( const double rad ) { return M_PI * rad * rad; } // PI * r ^ 2


static inline double alpha_m  ( const double v ) { return 0.32 * ( 13.1 - ( v - V_LEAK ) ) / ( exp ( ( 13.1 - ( v - V_LEAK ) ) / 4.0 ) - 1 ); }
static inline double alpha_s  ( const double v ) { return 1.6 / ( 1 + exp ( - 0.072 * ( ( v - V_LEAK ) - 65 ) ) ); }
static inline double alpha_n  ( const double v ) { return 0.016 * ( 35.1 - ( v - V_LEAK ) ) / ( exp ( ( 35.1 - ( v - V_LEAK ) ) / 5.0 ) - 1 ); }
static inline double alpha_c  ( const double v ) { return ( v <= ( 50 + V_LEAK ) ) ? exp ( ( ( v - V_LEAK ) - 10 ) / 11.0 - ( ( v - V_LEAK ) - 6.5 ) / 27.0 ) / 18.975 : 2 * exp ( - ( ( v - V_LEAK ) - 6.5 ) / 27.0 ); }
static inline double alpha_a  ( const double v ) { return 0.02 * ( 13.1 - ( v - V_LEAK ) ) / ( exp ( ( 13.1 - ( v - V_LEAK ) ) / 10.0 ) - 1 ); }
static inline double alpha_h  ( const double v ) { return 0.128 * exp ( ( 17 - ( v - V_LEAK ) ) / 18.0 ); }
static inline double alpha_r  ( const double v ) { return ( v <= ( 0 + V_LEAK ) ) ? 0.005 : exp ( - ( v - V_LEAK ) / 20.0 ) / 200.0; }
static inline double alpha_b  ( const double v ) { return 0.0016 * exp ( ( - 13 - ( v - V_LEAK ) ) / 18.0 ); }
static inline double alpha_q  ( const double x ) { return fmin ( ( 0.2e-4 ) * x, 0.01 ); }
static inline double beta_m   ( const double v ) { return 0.28 * ( ( v - V_LEAK ) - 40.1 ) / ( exp ( ( ( v - V_LEAK ) - 40.1 ) / 5.0 ) - 1 ); }
static inline double beta_s   ( const double v ) { return 0.02 * ( ( v - V_LEAK ) - 51.1 ) / ( exp ( ( ( v - V_LEAK ) - 51.1 ) / 5.0 ) - 1 ); }
static inline double beta_n   ( const double v ) { return 0.25 * exp ( ( 20 - ( v - V_LEAK ) ) / 40.0 ); }
static inline double beta_c   ( const double v ) { return ( v <= ( 50 + V_LEAK ) ) ? 2 * exp ( - ( ( v - V_LEAK ) - 6.5 ) / 27.0 ) - alpha_c ( v ) : 0; }
static inline double beta_a   ( const double v ) { return 0.0175 * ( ( v - V_LEAK ) - 40.1 ) / ( exp ( ( ( v - V_LEAK ) - 40.1 ) / 10.0 ) - 1 ); }
static inline double beta_h   ( const double v ) { return 4.0 / ( 1 + exp ( ( 40 - ( v - V_LEAK ) ) / 5.0 ) ); }
static inline double beta_r   ( const double v ) { return ( v <= ( 0 + V_LEAK ) ) ? 0 : 0.005 - alpha_r ( v ); }
static inline double beta_b   ( const double v ) { return 0.05 / ( 1 + exp ( ( 10.1 - ( v - V_LEAK ) ) / 5.0 ) ); }
static inline double beta_q   ( const double x ) { return 0.001; }

void initialize_traub ( double var [ N_VARS_TRAUB ] [ NC_TRAUB ], double I_inj [ NC_TRAUB ], double g_comp [ NC_TRAUB ] [ 2 ] );
void solve_euler_traub(double var[N_VARS_TRAUB][NC_TRAUB], double i_inj[NC_TRAUB], double g_comp[NC_TRAUB][2]);
