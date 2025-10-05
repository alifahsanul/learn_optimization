# Simulated Annealing

## Mathematical Foundation:

**Objective:** Minimize function f(x) where x is the solution vector

**Acceptance Probability:**
```
P(accept) = {
    1,                    if f(x_new) ≤ f(x_current)
    exp(-ΔE/T),          if f(x_new) > f(x_current)
}
```

Where:
- `ΔE = f(x_new) - f(x_current)` (energy difference)
- `T` = current temperature
- `f(x)` = objective function value

**Temperature Schedule:**
```
T(t+1) = α × T(t)
```
Where `α` is the cooling rate (typically 0.8-0.99)

## Algorithm Steps:

1. **Initialize:**
   - Start with solution x₀
   - Set initial temperature T₀
   - Set cooling rate α ∈ (0,1)

2. **Generate** neighbor solution:
   ```
   x_new = x_current + random_step()
   ```

3. **Calculate** energy difference:
   ```
   ΔE = f(x_new) - f(x_current)
   ```

4. **Accept** new solutions:
   - If ΔE ≤ 0 → accept immediately (better solution)
   - If ΔE > 0 → accept with probability P = exp(-ΔE/T)

5. **Cool down:**
   ```
   T = α × T
   ```

6. **Iterate** until T < T_min or max iterations reached

## Key Parameters:

- **Initial Temperature (T₀):** Controls initial exploration (higher = more random moves)
- **Cooling Rate (α):** Controls convergence speed (lower = slower cooling)
- **Minimum Temperature (T_min):** Stopping criterion
- **Step Size:** Controls neighborhood size for generating new solutions