#!/usr/bin/env bash
# LunarLander RHUKF sweep  (rhukf.py)
# 72 runs = horizon{5,7} x alpha{0.1,0.3} x r_init{1.0,1.5,2.0} x p_init{0.03,0.05,0.07} x seed{0,42}
# train_mode = filter (RHUKF only), batch = 256
#
# -e 안 씀: run 하나 죽어도 나머지 계속.
set -uo pipefail

# ============================================================
#  CONFIG
# ============================================================
PY="python3"
SCRIPT="rhukf.py"                 # 프로젝트 루트(~/projects/rhukf)에서 실행
ENV="LunarLander-v3"              # 등록된 id (v2 아님)
EPISODES=1500                     # 목표 episode 수 (timing/24h에 직결 — 조절)

# 모든 run 공통 고정 플래그. 여기서 모델링 선택을 정한다.
# (gamma는 LunarLander라 0.99로 켜둠. 나머지는 네 결론대로 조절/주석해제)
EXTRA_FLAGS=(
  --gamma 0.99
  # --measurement_mode pure_reward     # 또는 q_target
  # --filter_form covariance
  # --state_form absolute
  # --activation_fn tanh                # tanh/mish/silu/...
  # --target_update_mode hard --target_update_period 500
)
# ============================================================

BATCH=256
TRAIN_MODE="filter"

HORIZONS=(5 7)
ALPHAS=(0.1 0.3)
R_INITS=(1.0 1.5 2.0)
P_INITS=(0.03 0.05 0.07)
SEEDS=(0 42)

OUTROOT="sweep_lunarlander_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTROOT"

total=$(( ${#HORIZONS[@]} * ${#ALPHAS[@]} * ${#R_INITS[@]} * ${#P_INITS[@]} * ${#SEEDS[@]} ))
echo "=========================================="
echo " LunarLander RHUKF sweep   (rhukf.py)"
echo " total runs : $total"
echo " episodes   : $EPISODES   batch : $BATCH   train_mode : $TRAIN_MODE"
echo " log root   : $OUTROOT   (rhukf.py가 자체 결과폴더는 cwd에 생성)"
echo "=========================================="

i=0
start_all=$(date +%s)

for n in "${HORIZONS[@]}"; do
 for a in "${ALPHAS[@]}"; do
  for r in "${R_INITS[@]}"; do
   for p in "${P_INITS[@]}"; do
    for s in "${SEEDS[@]}"; do
      i=$((i+1))
      tag="lunar_a${a}_r${r}_p${p}_b${BATCH}_h${n}_s${s}"
      log="${OUTROOT}/${tag}.log"
      done_mark="${OUTROOT}/${tag}.done"

      # --- resume: 끝난 run 건너뜀 ---
      if [[ -f "$done_mark" ]]; then
        echo "[$i/$total] SKIP (done) $tag"
        continue
      fi

      echo "[$i/$total] RUN  $tag"
      t0=$(date +%s)

      "$PY" "$SCRIPT" \
        --env "$ENV" \
        --episodes "$EPISODES" \
        --batch "$BATCH" \
        --N_horizon "$n" \
        --alpha "$a" \
        --r_init "$r" \
        --p_init "$p" \
        --seed "$s" \
        --train_mode "$TRAIN_MODE" \
        "${EXTRA_FLAGS[@]}" \
        > "$log" 2>&1
      rc=$?

      t1=$(date +%s)
      if [[ $rc -eq 0 ]]; then
        touch "$done_mark"
        echo "      done in $((t1-t0))s"
      else
        echo "      FAILED (rc=$rc) -> $log"
        echo "        tail:"; tail -3 "$log" | sed 's/^/        /'
      fi
    done
   done
  done
 done
done

end_all=$(date +%s)
echo "=========================================="
echo " finished $total runs in $(( (end_all-start_all)/60 )) min ($((end_all-start_all))s)"
echo " logs: $OUTROOT/*.log"
echo "=========================================="
