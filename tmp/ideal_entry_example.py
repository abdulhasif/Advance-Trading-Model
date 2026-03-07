"""
Educational chart: What kind of uptrend does the Dual-Brain model CATCH?
Shows side-by-side:
  LEFT  → Today's ABB (rejected - choppy, retail-driven)
  RIGHT → Ideal "institutional" uptrend the model would catch
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ── ABB TODAY (Left Panel) ────────────────────────────────────────────────────
# Mimics: open dip → moderate rally → correction → afternoon spike → fade
t_abb = np.linspace(9.25, 15.5, 400)  # 9:15 AM to 3:30 PM in hours

def abb_today(t):
    """Simulate ABB's choppy W-pattern today"""
    base = 5833
    # Morning dip then recovery
    phase1 = 90 * (1 - np.exp(-2*(t-9.25))) * (t<11.5)    # morning rally  
    # Midday correction
    phase2 = -80 * np.exp(-((t-12.0)**2)/0.3) * (t>11.0) * (t<13.5)
    # Afternoon grind + spike
    phase3 = 40 * (t>13.0) * (t-13.0) + 30 * np.exp(-((t-14.7)**2)/0.05) * (t>14.0)
    # EOD fade
    phase4 = -70 * (t>14.8) * (1 - np.exp(-3*(t-14.8)))
    noise = np.random.randn(len(t)) * 5
    return base + phase1 + phase2 + phase3 + phase4 + noise

price_abb = abb_today(t_abb)

# Probability curve for ABB today (never crosses 0.55)
def prob_abb(t):
    p = 0.30 + 0.08*np.sin(3*(t-9.25)) + 0.05*np.abs(np.sin(5*t))
    return np.clip(p, 0.15, 0.48)  # Never crosses 0.55

prob_abb_vals = prob_abb(t_abb)

# ── IDEAL INSTITUTIONAL UPTREND (Right Panel) ─────────────────────────────────
# Characteristics: explosive start, sustained momentum, clean bricks, sector leader
t_ideal = np.linspace(9.25, 15.5, 400)

def ideal_uptrend(t):
    """Ideal scenario: explosive institutional breakout after 10 AM consolidation"""
    base = 5600
    # Morning consolidation (tight range)
    pre_break = 20 * np.sin(4*(t-9.25)) * (t < 10.5)
    # EXPLOSIVE BREAKOUT at ~10:30
    breakout = 300 * (1 - np.exp(-4*(t-10.5))) * (t >= 10.5) * (t < 12.5)
    # Mild pullback / healthy consolidation
    pullback = -40 * np.exp(-((t-12.7)**2)/0.15) * (t > 12.0)
    # Second leg up (sector momentum)
    second_leg = 80 * (1 - np.exp(-3*(t-13.0))) * (t >= 13.0) * (t < 14.5)
    # EOD orderly exit
    eod = -30 * (t > 14.8) * (t - 14.8)
    noise = np.random.randn(len(t)) * 3
    return base + pre_break + breakout + pullback + second_leg + eod + noise

price_ideal = ideal_uptrend(t_ideal)
price_ideal = np.clip(price_ideal, 5590, 5610 + 430)  # bound

# Probability curve for ideal (crosses 0.55 clearly)
def prob_ideal(t):
    # Low before breakout, sharp spike at entry, sustained above threshold
    pre = 0.30 + 0.03*np.sin(4*t)
    swell = 0.45 * (1 - np.exp(-6*(t-10.4))) * (t >= 10.4)
    decay = -0.15 * (1 - np.exp(-3*(t-13.5))) * (t >= 13.5)
    return np.clip(pre + swell + decay, 0.15, 0.85)

prob_ideal_vals = prob_ideal(t_ideal)

# ── RENKO BRICK SIMULATION ────────────────────────────────────────────────────
def simulate_renko(prices, brick_size=8.8):
    """Simulate simplified Renko bricks"""
    bricks = []
    last_close = prices[0]
    for p in prices:
        while p >= last_close + brick_size:
            bricks.append(('up', last_close, last_close + brick_size))
            last_close = last_close + brick_size
        while p <= last_close - brick_size:
            bricks.append(('down', last_close, last_close - brick_size))
            last_close = last_close - brick_size
    return bricks

# ── PLOTTING ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor='#0d1117')
fig.suptitle('DUAL-BRAIN: What Makes a Trade "Catchable"?', 
             fontsize=16, color='white', fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[3, 1.2, 0.3], hspace=0.4, wspace=0.12)

ax_abb_price  = fig.add_subplot(gs[0, 0])
ax_abb_prob   = fig.add_subplot(gs[1, 0], sharex=ax_abb_price)
ax_ideal_price = fig.add_subplot(gs[0, 1])
ax_ideal_prob  = fig.add_subplot(gs[1, 1], sharex=ax_ideal_price)

dark_bg = '#0d1117'
panel_bg = '#161b22'
for ax in [ax_abb_price, ax_abb_prob, ax_ideal_price, ax_ideal_prob]:
    ax.set_facecolor(panel_bg)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_color('#30363d')
    ax.spines['right'].set_color('#30363d')
    ax.grid(True, color='#21262d', alpha=0.8, linewidth=0.5)

def fmt_time(t):
    h = int(t)
    m = int((t - h)*60)
    return f"{h}:{m:02d}"

# ─── LEFT: ABB TODAY ──────────────────────────────────────────────────────────
ax_abb_price.set_title('❌  TODAY — ABB (REJECTED)', color='#f85149', 
                        fontsize=12, fontweight='bold', pad=8)
ax_abb_price.plot(t_abb, price_abb, color='#58a6ff', linewidth=1.2, alpha=0.8, label='Price')
ax_abb_price.set_ylabel('Price (₹)', color='#8b949e', fontsize=9)
ax_abb_price.yaxis.label.set_color('#8b949e')
ax_abb_price.tick_params(axis='x', labelbottom=False)

# Annotate key blockers
ax_abb_price.annotate('Choppy move\n(slow velocity)', 
                       xy=(11.0, 5940), xytext=(10.3, 5960),
                       fontsize=7.5, color='#f85149',
                       arrowprops=dict(arrowstyle='->', color='#f85149', lw=1),
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#f85149'))

ax_abb_price.annotate('Lagging sector\n(RS=-0.85)', 
                       xy=(9.8,  5870), xytext=(9.4, 5840),
                       fontsize=7.5, color='#f0883e',
                       arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1),
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#f0883e'))

ax_abb_price.annotate('EOD spike\n(NO_ENTRY_HOUR)', 
                       xy=(14.7, price_ideal.max()-60), xytext=(13.5, 5933),
                       fontsize=7.5, color='#8b949e',
                       arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1),
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#8b949e'))

# Probability panel - ABB
ax_abb_prob.plot(t_abb, prob_abb_vals, color='#3fb950', linewidth=1.5, label='P(LONG)')
ax_abb_prob.axhline(0.55, color='#f85149', linewidth=1.5, linestyle='--', label='Threshold 0.55')
ax_abb_prob.fill_between(t_abb, prob_abb_vals, 0.55, where=(prob_abb_vals>=0.55), 
                          color='#3fb950', alpha=0.3, label='Would trade zone')
ax_abb_prob.set_ylim(0, 1)
ax_abb_prob.set_ylabel('P(LONG)', color='#8b949e', fontsize=9)
ax_abb_prob.legend(fontsize=7, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')

# Never crosses - fill the "stuck below" area
ax_abb_prob.fill_between(t_abb, prob_abb_vals, 0, alpha=0.15, color='#8b949e')
ax_abb_prob.text(12.5, 0.60, '← Never crosses 0.55\n    (Too many blockers)', 
                  color='#f85149', fontsize=8, ha='center')

# X axis
tick_times = [9.25, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15.5]
ax_abb_prob.set_xticks(tick_times)
ax_abb_prob.set_xticklabels([fmt_time(t) for t in tick_times], rotation=30, fontsize=7)

# ─── RIGHT: IDEAL UPTREND ─────────────────────────────────────────────────────
ax_ideal_price.set_title('✅  IDEAL — High-Conviction Breakout (WOULD TRADE)', color='#3fb950', 
                          fontsize=12, fontweight='bold', pad=8)
ax_ideal_price.plot(t_ideal, price_ideal, color='#58a6ff', linewidth=1.2, alpha=0.8, label='Price')
ax_ideal_price.set_ylabel('Price (₹)', color='#8b949e', fontsize=9)
ax_ideal_price.tick_params(axis='x', labelbottom=False)

# Entry arrow
entry_t = 10.55
entry_p = ideal_uptrend(np.array([entry_t]))[0]

# Calculate approximate exit
exit_t = 13.0
exit_p = ideal_uptrend(np.array([exit_t]))[0]

ax_ideal_price.annotate('🟢 LONG ENTRY\n(P=0.74, RS=+1.2)', 
                          xy=(entry_t, entry_p), xytext=(entry_t+0.3, entry_p-80),
                          fontsize=8, color='#3fb950', fontweight='bold',
                          arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.5),
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d2b1a', edgecolor='#3fb950'))

ax_ideal_price.annotate('🔴 EXIT\n+2.8% PnL', 
                          xy=(exit_t, exit_p), xytext=(exit_t+0.3, exit_p+40),
                          fontsize=8, color='#f85149', fontweight='bold',
                          arrowprops=dict(arrowstyle='->', color='#f85149', lw=1.5),
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#2d0b0b', edgecolor='#f85149'))

ax_ideal_price.annotate('Explosive velocity\n(low wick pressure)', 
                          xy=(11.2, price_ideal[160]), xytext=(11.8, price_ideal[160]-60),
                          fontsize=7.5, color='#79c0ff',
                          arrowprops=dict(arrowstyle='->', color='#79c0ff', lw=1),
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#79c0ff'))

ax_ideal_price.annotate('Consolidation\n(healthy pullback)', 
                          xy=(12.7, price_ideal[225]), xytext=(13.3, price_ideal[225]-60),
                          fontsize=7.5, color='#e3b341',
                          arrowprops=dict(arrowstyle='->', color='#e3b341', lw=1),
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#e3b341'))

# Shade the trade zone
trade_mask = (t_ideal >= entry_t) & (t_ideal <= exit_t)
ax_ideal_price.fill_between(t_ideal, price_ideal, price_ideal.min()-10, 
                             where=trade_mask, alpha=0.12, color='#3fb950')

# Probability panel - ideal
ax_ideal_prob.plot(t_ideal, prob_ideal_vals, color='#3fb950', linewidth=1.5, label='P(LONG)')
ax_ideal_prob.axhline(0.55, color='#f85149', linewidth=1.5, linestyle='--', label='Threshold 0.55')
ax_ideal_prob.fill_between(t_ideal, prob_ideal_vals, 0.55, 
                            where=(prob_ideal_vals >= 0.55), 
                            color='#3fb950', alpha=0.25, label='Would trade zone')
ax_ideal_prob.set_ylim(0, 1)
ax_ideal_prob.set_ylabel('P(LONG)', color='#8b949e', fontsize=9)
ax_ideal_prob.legend(fontsize=7, facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
ax_ideal_prob.fill_between(t_ideal, prob_ideal_vals, 0, alpha=0.1, color='#3fb950')

# Entry/exit markers on prob
ax_ideal_prob.axvline(entry_t, color='#3fb950', linewidth=1, linestyle=':', alpha=0.7)
ax_ideal_prob.axvline(exit_t, color='#f85149', linewidth=1, linestyle=':', alpha=0.7)
ax_ideal_prob.text(entry_t+0.05, 0.80, '▶ ENTRY', color='#3fb950', fontsize=7)
ax_ideal_prob.text(exit_t+0.05, 0.80, '◀ EXIT', color='#f85149', fontsize=7)

ax_ideal_prob.set_xticks(tick_times)
ax_ideal_prob.set_xticklabels([fmt_time(t) for t in tick_times], rotation=30, fontsize=7)

# ── BOTTOM INFO PANEL ─────────────────────────────────────────────────────────
ax_info = fig.add_subplot(gs[2, :])
ax_info.set_facecolor('#161b22')
ax_info.axis('off')

info_text = (
    "  🔴  REJECTED  (ABB Today)  "
    "Slow velocity • Lagging sector (RS=-0.85) • High wick rejection • Random Walk regime (Hurst=0.50) • Peaked at P=0.45 only"
    "           |           "
    "  🟢  IDEAL SETUP  "
    "High velocity (explosive bricks) • Leading sector (RS>+0.5) • Clean wick (no rejection) • Trending regime (Hurst>0.55) • P>0.65 sustained"
)
ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=8.5,
             color='#c9d1d9', wrap=True,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#30363d'))

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "storage/logs/ideal_entry_example.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#0d1117')
print(f"Chart saved → {out}")
