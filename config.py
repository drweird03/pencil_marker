# ============================================================
# pencil_marker — tuning knobs
# Edit any value here, then press R in the app to reload.
# Keyboard shortcuts still work on top of these defaults.
# ============================================================

# ---- Grid --------------------------------------------------
CELL_SIZE      = 7      # grid cell size in pixels 2-20   ([ / ] keys ±2)
MAX_DOTS       = 27     # max dots per cell at full dark   (- / = keys ±5)
DOT_POWER      = 3      # curve exponent: dots = max * (1-brightness)^N
                        # 1=linear  2=quadratic  3=cubic   (O / P keys ±1)
DOT_PREFERENCE = 4.0    # TSP similarity bias               (, / . keys ±0.5)
                        # 0=pure nearest-neighbor  higher=strongly prefers same-brightness neighbors
SPLINE_N       = 10     # Catmull-Rom steps per segment    (higher = smoother line)
                        # 5=fast/rough  10=default  20=very smooth

# ---- Wanderer ----------------------------------------------
STEER_SCALE     = 10.0  # multiplier on raw steer value    ([ / ] keys ±0.05)
STEPS_PER_FRAME = 120   # physics steps per display frame  (more = faster coverage)
