#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <stdint.h>
#include <stdbool.h>

#include"traub.h"

const double g_Na    [ NC_TRAUB ] = { 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 15.0, 30.0, 15.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
const double g_K_DR  [ NC_TRAUB ] = { 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 5.0, 15.0, 5.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
const double g_K_A   [ NC_TRAUB ] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
const double g_K_C   [ NC_TRAUB ] = { 0.0, 5.0, 5.0, 10.0, 10.0, 10.0, 5.0, 20.0, 10.0, 20.0, 5.0, 15.0, 15.0, 15.0, 15.0, 15.0, 5.0, 5.0, 0.0 };
const double g_K_AHP [ NC_TRAUB ] = { 0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0 };
const double g_Ca    [ NC_TRAUB ] = { 0.0, 5.0, 5.0, 12.0, 12.0, 12.0, 5.0, 8.0, 4.0, 8.0, 5.0, 17.0, 17.0, 17.0, 10.0, 10.0, 5.0, 5.0, 0.0 };
const double g_leak  [ NC_TRAUB ] = { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
const double phi     [ NC_TRAUB ] = { 7769, 7769, 7769, 7769.0, 7769.0, 7769.0, 7769.0, 34530.0, 17402.0, 26404.0, 5941.0, 5941.0, 5941.0, 5941.0, 5941.0, 5941.0, 5941.0, 5941.0, 5941.0 };
const double rad     [ NC_TRAUB ] = { 2.89e-4, 2.89e-4, 2.89e-4, 2.89e-4, 2.89e-4, 2.89e-4, 2.89e-4, 2.89e-4, 4.23e-4, 2.42e-4, 2.42e-4, 2.42e-4, 2.42e-4, 2.42e-4, 2.42e-4, 2.42e-4, 2.42e-4, 2.42e-4, 2.42e-4 };
const double len     [ NC_TRAUB ] = { 1.20e-2, 1.20e-2, 1.20e-2, 1.20e-2, 1.20e-2, 1.20e-2, 1.20e-2, 1.20e-2, 1.25e-2, 1.10e-2, 1.10e-2, 1.10e-2, 1.10e-2, 1.10e-2, 1.10e-2, 1.10e-2, 1.10e-2, 1.10e-2, 1.10e-2 };
const double area    [ NC_TRAUB ] = { 2.188e-5, 2.188e-5, 2.188e-5, 2.188e-5, 2.188e-5, 2.188e-5, 2.188e-5, 2.188e-5, 3.320e-5, 1.673e-5, 1.673e-5, 1.673e-5, 1.673e-5, 1.673e-5, 1.673e-5, 1.673e-5, 1.673e-5, 1.673e-5, 1.673e-5 };


static inline double tau ( const int var, const double v )
{
  switch ( var ) {
  case M:
    return 1.0 / ( alpha_m ( v ) + beta_m ( v ) );
  case S:
    return 1.0 / ( alpha_s ( v ) + beta_s ( v ) );
  case N:
    return 1.0 / ( alpha_n ( v ) + beta_n ( v ) );
  case C:
    return 1.0 / ( alpha_c ( v ) + beta_c ( v ) );
  case A:
    return 1.0 / ( alpha_a ( v ) + beta_a ( v ) );
  case H:
    return 1.0 / ( alpha_h ( v ) + beta_h ( v ) );
  case R:
    return 1.0 / ( alpha_r ( v ) + beta_r ( v ) );
  case B:
    return 1.0 / ( alpha_b ( v ) + beta_b ( v ) );
  case Q:
    return 1.0 / ( alpha_q ( v ) + beta_q ( v ) );
  default:
    fprintf ( stderr, "Error: no such vars: %d\n", var );
    exit ( 1 );
  }
}

static inline double inf ( const int var, const double v )
{
  switch ( var ) {
  case M:
    return alpha_m ( v ) / ( alpha_m ( v ) + beta_m ( v ) );
  case S:
    return alpha_s ( v ) / ( alpha_s ( v ) + beta_s ( v ) );
  case N:
    return alpha_n ( v ) / ( alpha_n ( v ) + beta_n ( v ) );
  case C:
    return alpha_c ( v ) / ( alpha_c ( v ) + beta_c ( v ) );
  case A:
    return alpha_a ( v ) / ( alpha_a ( v ) + beta_a ( v ) );
  case H:
    return alpha_h ( v ) / ( alpha_h ( v ) + beta_h ( v ) );
  case R:
    return alpha_r ( v ) / ( alpha_r ( v ) + beta_r ( v ) );
  case B:
    return alpha_b ( v ) / ( alpha_b ( v ) + beta_b ( v ) );
  case Q:
    return alpha_q ( v ) / ( alpha_q ( v ) + beta_q ( v ) );
  default:
    fprintf ( stderr, "Error: no such vars: %d\n", var );
    exit ( 1 );
  }
}

void initialize_traub ( double var [ N_VARS_TRAUB ] [ NC_TRAUB ], double I_inj [ NC_TRAUB ], double g_comp [ NC_TRAUB ] [ 2 ] )
{
  // Membrane potential
  for ( int32_t i = 0; i < NC_TRAUB; i++ ) {
    var [ V ] [ i ] = V_LEAK;
    I_inj [ i ] = 0.;
  }

  // Gate variables
  for ( int32_t i = 0; i < NC_TRAUB; i++ ) {
    for ( int32_t j = 2; j < N_VARS_TRAUB - 1; j++ ) {
      var [ j ] [ i ] = inf ( j, var [ V ] [ i ] );
    }
  }
  for ( int32_t i = 0; i < NC_TRAUB; i++ ) {
    double i_Ca = g_Ca [ i ] * area [ i ] * var [ S ] [ i ] * var [ S ] [ i ] * var [ R ] [ i ] * ( var [ V ] [ i ] - V_Ca );
    var [ XI ] [ i ] = - i_Ca * phi [ i ] / Beta;
    var [ Q  ] [ i ] = inf ( Q, var [ XI ] [ i ] );
  }

  // Conductance between compartments
  for ( int32_t i = 0; i < NC_TRAUB; i++ ) {
    g_comp [ i ] [ 0 ] = ( i == 0      ) ? 0. : 2.0 / ( ( Ri * len [ i - 1 ] ) / pir2 ( rad [ i - 1 ] ) + ( Ri * len [ i     ] ) / pir2 ( rad [ i     ] ) );
    g_comp [ i ] [ 1 ] = ( i == NC_TRAUB - 1 ) ? 0. : 2.0 / ( ( Ri * len [ i     ] ) / pir2 ( rad [ i     ] ) + ( Ri * len [ i + 1 ] ) / pir2 ( rad [ i + 1 ] ) );
  }
}

void solve_euler_traub ( double var [ N_VARS_TRAUB ] [ NC_TRAUB ], double i_inj [ NC_TRAUB ], double g_comp [ NC_TRAUB ] [ 2 ] )
{
  double dvar [ N_VARS_TRAUB ] [ NC_TRAUB ];
  double i_ion [ NC_TRAUB ], i_comp [ NC_TRAUB ];

  for ( int32_t i = 0; i < NC_TRAUB; i++ ) {
    double *v = var [ V ];
    for ( int32_t j = 2; j < N_VARS_TRAUB - 1; j++ ) {
      dvar [ j ] [ i ] = ( DT / tau ( j, v [ i ] ) ) * ( - var [ j ] [ i ] + inf ( j, v [ i ] ) );
    }
    dvar [ Q ] [ i ] = ( DT / tau ( Q, var [ XI ] [ i ] ) ) * ( - var [ Q ] [ i ] + inf ( Q, var [ XI ] [ i ] ) );

    i_ion [ i ] = ( g_leak    [ i ] * ( v [ i ] - V_LEAK )
                    + g_Na    [ i ] * var [ M ] [ i ] * var [ M ] [ i ] * var [ H ] [ i ] * ( v [ i ] - V_Na )
		    + g_Ca    [ i ] * var [ S ] [ i ] * var [ S ] [ i ] * var [ R ] [ i ] * ( v [ i ] - V_Ca )
		    + g_K_DR  [ i ] * var [ N ] [ i ] * ( v [ i ] - V_K )
		    + g_K_A   [ i ] * var [ A ] [ i ] * var [ B ] [ i ] * ( v [ i ] - V_K )
		    + g_K_AHP [ i ] * var [ Q ] [ i ] * ( v [ i ] - V_K )
		    + g_K_C   [ i ] * var [ C ] [ i ] * fmin ( 1, var [ XI ] [ i ] / 250.0 ) * ( v [ i ] - V_K ) );

    i_comp [ i ] = (   ( ( i == 0      ) ? 0. : g_comp [ i ] [0] * ( v [ i - 1 ] - v [ i ] ) / area [ i ] )
		     + ( ( i == NC_TRAUB - 1 ) ? 0. : g_comp [ i ] [1] * ( v [ i + 1 ] - v [ i ] ) / area [ i ] ) );
    
    dvar [ V ] [ i ] = ( DT / Cm ) * ( - i_ion [ i ] + i_comp [ i ] + i_inj [ i ] );

    double i_Ca = g_Ca [ i ] * area [ i ] * var [ S ] [ i ] * var [ S ] [ i ] * var [ R ] [ i ] * ( var [ V ] [ i ] - V_Ca );
    dvar [ XI ] [ i ] = DT * ( - phi [ i ] * i_Ca - Beta * var [ XI ] [ i ] );
  }

  for ( int32_t i = 0; i < NC_TRAUB; i++ ) {
    for ( int32_t j = 0; j < N_VARS_TRAUB; j++ ) {
      var [ j ] [ i ] += dvar [ j ] [ i ];
    }
  }
}
