'''
Group 39 MAT292 project code
Code summary:
    - implement the vanilla Black-Scholes 
    - implement LSM Monte Carlo
    - Choose a contract from the csv
    - calibrate the volatility
    - make a plot comparing market vs BS vs LSM
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar


def bs_price_euro(S, K, r, sigma, T, is_put=False):
    """
        S = stocki price at valuation date (we chose the closing price of that market day)
        K = strike price
        r = risk-free rate (annualized, continuously compounded)
        sigma = constant volatility (annualized)
        T = time to maturity in years
        is_put = True for put, False for call
    """

    # cast to float just in case
    S = float(S); K = float(K); r = float(r); sigma = float(sigma); T = float(T)
    if T <= 0 or sigma <= 0: #if thhe option expired or no volatility 
        intrinsic = max(K - S, 0.0) if is_put else max(S - K, 0.0)
        return float(intrinsic)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    #look at report paper for formula
    if is_put:
        return float(K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))
    else:
        return float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))


def _laguerre_vander(x, deg):
    # build laguerre polynomial for regression basis
    from numpy.polynomial.laguerre import lagvander
    return lagvander(x, deg)

def lsm_american_gbm_cv(
    S0, K, r, sigma, T, is_put=True,
    steps=55, paths=35_000, seed=0,
    basis_deg=3, min_itm=350, ridge_base=1e-8
):
    """
    algorithm:
      - simulate many GBM path
      - backward induction
      - at each time step, estimate continuation value by regressing future cashflows on basis functions of S_t
      - exercise if intrinsic >= estimated continuation value
    """

    rng = np.random.default_rng(seed)

    dt = T / steps
    disc = np.exp(-r * dt) #discount

    # Antithetic variates setup 
    half = max(1, paths // 2)
    paths = 2 * half # make even
    path = np.empty((steps + 1, paths), dtype=float)

    S = np.full(paths, float(S0), dtype=float)
    path[0] = S

    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    for t in range(steps):
        Z = rng.standard_normal(half) # draw half normals
        Z = np.concatenate([Z, -Z]) # antithetic
        Z = (Z - Z.mean()) / (Z.std(ddof=1) + 1e-12) # moment matching
        S = S * np.exp(drift + vol * Z) # evolve all paths one step
        path[t + 1] = S

    # G_{i,n} for intrinsic price
    # put: max(K - S, 0) and Call: max(S - K, 0)
    intrinsic = np.maximum(K - path, 0.0) if is_put else np.maximum(path - K, 0.0)
    cf = intrinsic[-1].copy()

    # backward induction (LSM)
    for t in range(steps - 1, 0, -1):

        itm = intrinsic[t] > 0.0
        n_itm = int(np.sum(itm))

        if n_itm < min_itm:
            cf *= disc
            continue

        X = path[t, itm] # X = S_t values on ITM paths
        Y = cf[itm] * disc # discounted future cashflows back one step
        x = np.maximum(X / K, 1e-12)
        A = _laguerre_vander(x, basis_deg)

        lam = ridge_base * (basis_deg + 1) * (paths / max(n_itm, 1))**0.5

        # method of regression var: beta = argmin ||A beta - Y||^2 + lam ||beta||^2 
        ATA = A.T @ A
        ATY = A.T @ Y
        beta = np.linalg.solve(ATA + lam * np.eye(ATA.shape[0]), ATY)
        cont = A @ beta
        ex_val = intrinsic[t, itm]

        exercise = ex_val >= cont

        idx = np.where(itm)[0][exercise]
        cf[idx] = ex_val[exercise]
        cf *= disc

    X_am = cf.copy() 

    # control variate using vanilla BS price
    Y_euro = intrinsic[-1] * np.exp(-r * T)
    bs_euro = bs_price_euro(S0, K, r, sigma, T, is_put=is_put)
    vy = float(np.var(Y_euro, ddof=1))
    if vy > 0:
        cov = float(np.cov(X_am, Y_euro, ddof=1)[0, 1])
        b = cov / vy
    else:
        b = 0.0
    # variate adjusted estimator
    X_cv = X_am + b * (bs_euro - Y_euro)

    price_cv = float(np.mean(X_cv))
    se_cv = float(np.std(X_cv, ddof=1) / np.sqrt(paths))
    return price_cv, se_cv


# csv loading n market price choice


def _pick_contract_column(df):
    #dont change the csv names
    for c in ["contract", "contractSymbol", "symbol", "option_symbol", "OptionSymbol"]:
        if c in df.columns:
            return c
    raise KeyError(f"No contract id column found. Columns={list(df.columns)}")

opt = pd.read_csv("options_history_all.csv", parse_dates=["date"])
ul  = pd.read_csv("underlying_prices.csv", parse_dates=["date"])
opt["underlying"] = opt["underlying"].astype(str).str.upper().str.strip()
ul["underlying"]  = ul["underlying"].astype(str).str.upper().str.strip()

contract_col = _pick_contract_column(opt)

opt[contract_col] = opt[contract_col].astype(str).str.upper().str.strip()

opt["right"] = opt["right"].astype(str).str.upper().str.strip()
opt["expiration"] = pd.to_datetime(opt["expiration"])
opt["strike"] = pd.to_numeric(opt["strike"], errors="coerce")

for c in ["bid", "ask", "close", "volume", "openInterest"]:
    if c in opt.columns:
        opt[c] = pd.to_numeric(opt[c], errors="coerce")


ul["S"] = pd.to_numeric(ul["S"], errors="coerce")
df = opt.merge(ul[["date", "underlying", "S"]], on=["date", "underlying"], how="left")
df = df.dropna(subset=["S", "expiration", "strike", "right", contract_col]).copy()
df["T_years"] = (df["expiration"] - df["date"]).dt.days.clip(lower=0) / 365.0
df = df[df["T_years"] > 0].copy() # only keep rows with positive time to maturity


if "bid" in df.columns and "ask" in df.columns:
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df = df.dropna(subset=["mid", "bid", "ask"]).copy()
    df = df[df["ask"] >= df["bid"]]
    spread = (df["ask"] - df["bid"])
    df = df[spread / df["mid"].clip(lower=1e-6) < 0.35]

    price_col = "mid"
    price_label = "Market mid (CSV)"
else:
    # if no bid/ask -> just use close price
    df = df.dropna(subset=["close"]).copy()
    price_col = "close"
    price_label = "Market close (CSV)"



# pick contract n calinrate sigma: match put volatility with call volatility 

r = 0.05
SYMS = ["TSLA", "TLT", "JNJ", "UNH"]

# runtime/plot controls
MAX_POINTS = 140         
DOT = 2                  
LSM_STEPS = 55           
LSM_PATHS = 35_000       
BASIS_DEG = 3            
MIN_ITM = 350            
RIDGE_BASE = 1e-8        
BASE_SEED = 1234         
TRAIN_FRAC = 0.60        

def pick_best_contract(df_sym_right):
    counts = df_sym_right.groupby(contract_col).size().sort_values(ascending=False)
    for c in counts.index[:40]:
        d = df_sym_right[df_sym_right[contract_col] == c]
        if d["strike"].nunique() == 1 and d["expiration"].nunique() == 1:
            return c
    return counts.index[0] if len(counts) else None

def calibrate_sigma_constant_contract(d_contract, is_put, train_frac=0.6):
    d = d_contract.sort_values("date").copy()
    n = len(d)

    n_train = max(12, int(np.floor(n * train_frac)))
    train = d.iloc[:n_train].copy()

    S = train["S"].astype(float).values
    K = float(train["strike"].iloc[0]) # strike is constant for a contract
    T = train["T_years"].astype(float).values
    mkt = train[price_col].astype(float).values # market price series

    mask = (S > 0) & (K > 0) & (T > 0) & np.isfinite(mkt) & (mkt > 0)
    S, T, mkt = S[mask], T[mask], mkt[mask]

    if len(mkt) < 12:
        return 0.30
    
    def mse(sig):
        sig = float(sig)

        d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)

        if is_put:
            bs = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        else:
            bs = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

        return float(np.mean((bs - mkt) ** 2))
    res = minimize_scalar(mse, bounds=(0.01, 3.0), method="bounded")

    if (not res.success) or (not np.isfinite(res.x)):
        return 0.30
    return float(np.clip(res.x, 0.01, 3.0))

# match put with calls with the same expiry, closest strike price, densest history
def find_matching_call_series(sym, put_strike, put_exp):

    d_calls = df[
        (df["underlying"] == sym) &
        (df["right"] == "C") &
        (df["expiration"] == put_exp)
    ].copy()

    if d_calls.empty:
        return None


    cand = (d_calls.groupby(contract_col)
                  .agg(strike=("strike", "first"), n=("strike","size"))
                  .reset_index())

    cand["dist"] = (cand["strike"] - put_strike).abs()
    cand = cand.sort_values(["dist", "n"], ascending=[True, False]).head(10)
    best_contract = cand.sort_values(["dist","n"], ascending=[True, False])[contract_col].iloc[0]
    d_call = d_calls[d_calls[contract_col] == best_contract].sort_values("date").copy()

    if d_call["strike"].nunique() != 1 or d_call["expiration"].nunique() != 1:
        return None

    return d_call


# build and show one plot for a symbol + right


def plot_one(sym, right):

    d_all = df[(df["underlying"] == sym) & (df["right"] == right)].copy()
    if d_all.empty:
        print(f"[WARN] No rows for {sym} {right}")
        return

    contract = pick_best_contract(d_all)
    if contract is None:
        print(f"[WARN] Could not pick contract for {sym} {right}")
        return

    d = d_all[d_all[contract_col] == contract].sort_values("date").copy()

    if d["strike"].nunique() != 1 or d["expiration"].nunique() != 1:
        print(f"[WARN] Mixed series for {sym} {right} contract={contract} (skipping)")
        return


    if len(d) > MAX_POINTS:
        idx = np.linspace(0, len(d)-1, MAX_POINTS).round().astype(int)
        d = d.iloc[idx].copy()

    K = float(d["strike"].iloc[0])
    exp = pd.to_datetime(d["expiration"].iloc[0])
    is_put = (right == "P")
    sigma_source = "self"

    if is_put:
        d_call = find_matching_call_series(sym, put_strike=K, put_exp=exp)

        if d_call is not None and len(d_call) >= 20:
            sigma_hat = calibrate_sigma_constant_contract(d_call, is_put=False, train_frac=TRAIN_FRAC)
            sigma_source = f"matched CALL (K≈{float(d_call['strike'].iloc[0]):.2f})"
        else:
            # if no good call match, calibrate sigma on the put itself
            sigma_hat = calibrate_sigma_constant_contract(d, is_put=True, train_frac=TRAIN_FRAC)
            sigma_source = "PUT self (fallback)"
    else:
        sigma_hat = calibrate_sigma_constant_contract(d, is_put=False, train_frac=TRAIN_FRAC)
        sigma_source = "CALL self"

    bs_vals, lsm_vals, lsm_se = [], [], []

    # market prices and x-axis values (time to maturity)
    mkt = d[price_col].astype(float).values
    xT  = d["T_years"].astype(float).values

    # loop over each date-row in the contract time series
    for j, row in enumerate(d.itertuples(index=False)):
        S0 = float(row.S)
        T  = float(row.T_years)
        bs = bs_price_euro(S0, K, r, sigma_hat, T, is_put=is_put)
        lsm, se = lsm_american_gbm_cv(
            S0=S0, K=K, r=r, sigma=sigma_hat, T=T, is_put=is_put,
            steps=LSM_STEPS, paths=LSM_PATHS,
            seed=BASE_SEED + j + (0 if is_put else 10_000),
            basis_deg=BASIS_DEG, min_itm=MIN_ITM, ridge_base=RIDGE_BASE
        )

        bs_vals.append(bs)
        lsm_vals.append(lsm)
        lsm_se.append(se)

    bs_vals = np.array(bs_vals)
    lsm_vals = np.array(lsm_vals)
    lsm_se = np.array(lsm_se)

    # compute normalized RMSEs
    nrmse_bs = float(np.sqrt(np.mean((bs_vals - mkt)**2)))/(mkt.max() - mkt.min())
    nrmse_lsm = float(np.sqrt(np.mean((lsm_vals - mkt)**2)))/(mkt.max() - mkt.min())

    # plot code
    plt.figure(figsize=(10, 5))
    plt.plot(xT, mkt, marker="o", markersize=DOT, linewidth=1.6, label=price_label)
    plt.plot(xT, bs_vals, marker="o", markersize=DOT, linewidth=1.6, label="European BS (const σ)")
    plt.plot(xT, lsm_vals, marker="o", markersize=DOT, linewidth=1.6, label="LSM American (same const σ)")
    plt.fill_between(xT, lsm_vals - lsm_se, lsm_vals + lsm_se, alpha=0.18, label="LSM ±1 SE")
    plt.gca().invert_xaxis() # time matures to the right
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.xlabel("Time to maturity T (years)   (→ 0 = expiry)", fontsize = 13)
    plt.ylabel("Option price", fontsize = 13)

    # title with summary info
    kind = "PUT" if is_put else "CALL"
    plt.title(
        f"{sym} {kind} | K={K:.2f} | exp={exp.date()} | σ̂={sigma_hat*100:.2f}% "
        f"| NRMSE(BS)={nrmse_bs:.4f} NRMSE(LSM)={nrmse_lsm:.4f}",
        fontsize=14
    )

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # print summary in console 
    print(f"{sym} {right} contract={contract} | sigma_hat={sigma_hat:.4f} ({sigma_source}) | NRMSE(BS)={nrmse_bs:.4f} NRMSE(LSM)={nrmse_lsm:.4f}")

# main
for sym in SYMS:
    for right in ["C", "P"]:  # run the plots in both calls and puts
        plot_one(sym, right)