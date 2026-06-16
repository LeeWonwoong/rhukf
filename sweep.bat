@echo off

for %%N in (4 5 6) do (
    for %%Q in (1e-2 5e-3 1e-3) do (
        for %%A in (0.1 0.3 0.5 0.7 0.9) do (
            for %%R in (1.0 2.0 3.0) do (

                echo ==================================================
                echo N_horizon=%%N Q_std=%%Q alpha=%%A R_std=%%R
                echo ==================================================

                python rhukf.py ^
                    --N_horizon %%N ^
                    --alpha %%A ^
                    --q_init %%Q ^
                    --q_end %%Q ^
                    --r_init %%R ^
                    --r_end %%R

            )
        )
    )
)

pause