import jax

# HH ゲート変数（m/h/n）は exp を含む非線形 ODE のため float32 では数値不安定になる。
# compute_theta / _dummy_theta も float64 を明示要求しており、意図通りに動かすために必須。
jax.config.update("jax_enable_x64", True)
