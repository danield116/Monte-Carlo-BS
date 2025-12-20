'''
Notes for TA/Profs:
this script builds a reproducible dataset of option contract histories from yfinance
to reproduce the exact same graph as the report, please do not use this script.
Reason being that some contracts may have expired since the csv in the git repo was built,
and yfinance can't return expired contracts.
Please use the csv files in the repo directly for reproducing the report results.
However, feel free to use this script to build another dataset for anything else.

Regards,
Group 39.
'''


from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent



@dataclass(frozen=True)
class SelectionConfig:
    underlyings: Tuple[str, ...] = ("TSLA", "TLT", "JNJ", "UNH")
    target_dtes: Tuple[int, ...] = (14, 30, 60, 120)
    moneyness: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)
    max_contracts_per_underlying: int = 60
    start: str = "2024-01-01"
    end: Optional[str] = "2025-12-15" 
    sleep_s: float = 0.35


# helper functions

def nearest_value(values: np.ndarray, target: float) -> float:
    if values.size == 0:
        raise ValueError("Empty strike list.")
    return float(values[np.argmin(np.abs(values - target))])


def pick_expiration_by_dte(expirations, target_dte, today: pd.Timestamp) -> Optional[str]:
    if not expirations:
        return None

    exp_dt = pd.to_datetime(expirations, errors="coerce")
    exp_dt = exp_dt[~pd.isna(exp_dt)]
    if len(exp_dt) == 0:
        return None

    dtes = (exp_dt - today).days

    mask = dtes >= 0
    exp_dt = exp_dt[mask]
    dtes = dtes[mask]

    if len(exp_dt) == 0:
        return None

    idx = int(np.argmin(np.abs(dtes - target_dte)))

    return exp_dt[idx].strftime("%Y-%m-%d")


def safe_history(ticker: yf.Ticker, start: str, end: Optional[str]) -> pd.DataFrame:
    try:
        df = ticker.history(start=start, end=end, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep].copy()
    except Exception:
        return pd.DataFrame()


# select contracts

def select_contracts_for_underlying(sym: str, cfg: SelectionConfig) -> pd.DataFrame:
    '''
    Select a reproducible set of option contracts for one underlying, based on:
      - DTE targets (14/30/60/120)
      - moneyness targets (ATM, 5%, 10%, 20% away)
    '''
    t = yf.Ticker(sym)
    expirations = list(getattr(t, "options", []) or [])
    if not expirations:
        return pd.DataFrame(columns=["underlying", "expiration", "right", "strike", "contractSymbol"])

    today = pd.to_datetime(cfg.end).normalize() if cfg.end else pd.Timestamp.today().normalize()

    spot_window_start = (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    spot_hist = safe_history(t, start=spot_window_start, end=None)
    if spot_hist.empty:
        raise RuntimeError(f"Could not fetch underlying history for {sym}")
    spot = float(spot_hist["Close"].iloc[-1])

    picks = []
    seen = set()

    for target_dte in cfg.target_dtes:
        exp = pick_expiration_by_dte(expirations, target_dte, today=today)
        if exp is None:
            continue

        try:
            chain = t.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
        except Exception:
            continue

        if calls.empty or puts.empty:
            continue

        call_strikes = calls["strike"].to_numpy(dtype=float)
        put_strikes = puts["strike"].to_numpy(dtype=float)

        for m in cfg.moneyness:
            targets = [spot * (1.0 + m)]
            if m > 0:
                targets.append(spot * (1.0 - m))

            for k_target in targets:
                try:
                    k_call = nearest_value(call_strikes, k_target)
                    k_put = nearest_value(put_strikes, k_target)

                    c_row = calls.loc[calls["strike"] == k_call].iloc[0]
                    p_row = puts.loc[puts["strike"] == k_put].iloc[0]

                    c_sym = str(c_row["contractSymbol"])
                    p_sym = str(p_row["contractSymbol"])

                    key_c = (sym, exp, "C", float(k_call), c_sym)
                    key_p = (sym, exp, "P", float(k_put), p_sym)

                    if key_c not in seen:
                        seen.add(key_c)
                        picks.append(
                            dict(underlying=sym, expiration=exp, right="C", strike=float(k_call), contractSymbol=c_sym)
                        )

                    if key_p not in seen:
                        seen.add(key_p)
                        picks.append(
                            dict(underlying=sym, expiration=exp, right="P", strike=float(k_put), contractSymbol=p_sym)
                        )

                except Exception:
                    continue

    df = pd.DataFrame(picks)
    if df.empty:
        return df

    df["spot"] = spot
    df["abs_moneyness"] = np.abs(df["strike"] / df["spot"] - 1.0)
    df = df.sort_values(["expiration", "abs_moneyness", "right"]).head(cfg.max_contracts_per_underlying)
    df = df.drop(columns=["spot", "abs_moneyness"])
    return df


def download_contract_histories(selected: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    meta = selected.set_index("contractSymbol")[["underlying", "expiration", "right", "strike"]].to_dict("index")
    contracts = selected["contractSymbol"].drop_duplicates().tolist()

    out_parts = []

    for i, contract in enumerate(contracts, 1):
        print(f"[{i}/{len(contracts)}] {contract}")
        tk = yf.Ticker(contract)

        hist = safe_history(tk, start=cfg.start, end=cfg.end)
        time.sleep(cfg.sleep_s)

        if hist.empty:
            continue

        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]
        hist = hist.rename(columns={date_col: "date"})

        m = meta.get(contract, {})
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(hist["date"]).dt.tz_localize(None),
                "contract": contract,
                "underlying": m.get("underlying"),
                "expiration": m.get("expiration"),
                "right": m.get("right"),
                "strike": m.get("strike"),
                "open": pd.to_numeric(hist.get("Open", np.nan), errors="coerce"),
                "high": pd.to_numeric(hist.get("High", np.nan), errors="coerce"),
                "low": pd.to_numeric(hist.get("Low", np.nan), errors="coerce"),
                "close": pd.to_numeric(hist.get("Close", np.nan), errors="coerce"),
                "volume": pd.to_numeric(hist.get("Volume", np.nan), errors="coerce"),
            }
        )
        out_parts.append(part)

    if not out_parts:
        return pd.DataFrame(
            columns=["date","contract","underlying","expiration","right","strike","open","high","low","close","volume"]
        )

    df_all = pd.concat(out_parts, ignore_index=True)
    df_all = df_all.sort_values(["underlying", "contract", "date"])
    return df_all

def build_dataset(
    cfg: SelectionConfig,
    selected_csv=BASE_DIR / "selected_contracts.csv",
    history_csv=BASE_DIR / "options_history_all.csv",
    metadata_json=BASE_DIR / "dataset_metadata.json",
) -> None:

    selected_parts = []

    for sym in cfg.underlyings:
        print(f"\nSelecting contracts for {sym} ...")
        sel = select_contracts_for_underlying(sym, cfg)
        print(f"  selected: {len(sel)}")
        if not sel.empty:
            selected_parts.append(sel)

    if not selected_parts:
        raise RuntimeError("No contracts selected for any underlying.")

    selected = pd.concat(selected_parts, ignore_index=True)
    selected.to_csv(selected_csv, index=False)
    print(f"\nSaved -> {selected_csv} ({len(selected)} rows)")

    history = download_contract_histories(selected, cfg)
    history.to_csv(history_csv, index=False)
    print(f"Saved -> {history_csv} ({len(history)} rows)")

    meta = asdict(cfg)
    meta["built_at"] = pd.Timestamp.now().isoformat()
    meta["num_selected_contracts"] = int(selected["contractSymbol"].nunique())
    meta["num_history_rows"] = int(len(history))

    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved -> {metadata_json}")


if __name__ == "__main__":
    cfg = SelectionConfig(
        underlyings=("TSLA","TLT","JNJ","UNH"),
        target_dtes=(14, 30, 60, 120),
        moneyness=(0.00, 0.05, 0.10, 0.20),
        max_contracts_per_underlying=60,
        start="2024-01-01",
        end="2025-12-15",   # keep fixed for reproducibility
        sleep_s=0.35,
    )
    build_dataset(cfg)
