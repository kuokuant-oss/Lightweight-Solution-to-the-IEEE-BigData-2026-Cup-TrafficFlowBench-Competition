"""Feature rows for the served correction (one panel in memory at a time).
  s3_build_rows.py pseudo private [panels]   private pseudo-hole table, exactly round2/task1_trans/build_ph.py's validation
                                            path with split=private: pseudo windows = corrected selector on
                                            work/cands_private.parquet; drop windows whose [T-12, T+31) touches an official
                                            private hole; history T-12..T-1 target cells filled with the PARENT private
                                            state; truth = released observed value -> work/ph_private_<panel>.parquet
  s3_build_rows.py official <split> [panels] every eligible cell of the 80 official holes of the split, build_ph feature
                                            path (as round2 verify/off_apply.py) + timestamp/link_id
                                            -> work/off_rows_<split>.parquet"""
# Task 1 reproduction pipeline, step 5 (`pipeline/task1.sh`, logs `ph_private`
# and `off_rows_<split>`). Rows have the columns of
# round2/task1_trans/build_ph.py; official rows have no truth columns and add
# timestamp, link_id and window_id.
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
HERE = Path(__file__).resolve().parent
PILOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260925/round2/task1_trans')
sys.path.insert(0, str(PILOT))
import build_ph as B                     # read-only import (PYTHONDONTWRITEBYTECODE=1)
from build_ph import (R, grid, physics, P1, template_from_hist, LG, HISTN, block_ids, gap_matrix, SplitStats,
                      PANELS, MD, G2W, G2V2, KEEP_F)
from s2_cands import parent_state


def load_masked(panel, split):
    """Returns the masked day grids of a panel and split, sorted by date."""
    grids = [grid.build_masked(panel, p) for p in P1.split_masked_files(panel, split)]
    order = np.argsort([g.date for g in grids], kind='stable')
    return [grids[i] for i in order]


def models():
    """Returns (feature list, speed booster, flow booster) of the G2v2 v2_F hole model."""
    feats = json.load(open(MD / 'gap_features.json'))
    return feats, lgb.Booster(model_file=str(MD / 'gap_speed.txt')), lgb.Booster(model_file=str(MD / 'gap_flow.txt'))


def g2v2_predict(ms, mf, feats, f, panel, lanes):
    """Predicts hole speed and flow with G2v2 v2_F.

    Args:
      ms: speed booster.
      mf: flow-per-lane booster.
      feats: feature list.
      f: dict of (L, LG) feature arrays from gap_matrix.
      panel: corridor name (categorical feature panel_c).
      lanes: lanes per link.

    Returns:
      (pv, pq): (L, LG) speed in km/h (within the speed floor and cap) and
      flow in veh/h (at least the flow floor).
    """
    X = pd.DataFrame({k: np.asarray(v).reshape(-1).astype(np.float32) for k, v in f.items()})
    X['panel_c'] = pd.Categorical([panel] * len(X), categories=PANELS)
    pv = np.clip(ms.predict(X[feats], num_threads=3), P1.SPEED_FLOOR, P1.SPEED_CAP).reshape(f['step'].shape)
    pq = np.maximum(mf.predict(X[feats], num_threads=3) * np.repeat(lanes[:, None], LG, 1).reshape(-1), P1.FLOW_FLOOR).reshape(f['step'].shape)
    return pv, pq


def panel_ctx(panel, split):
    """Returns the per-panel context of a split.

    Day grids, link order and parameters, queue thresholds, lanes, blocks,
    train weekday profile, static queued-speed templates, split statistics
    and the official window index.
    """
    grids = load_masked(panel, split)
    links = [str(x) for x in grids[0].links]; times0 = [pd.Timestamp(t).strftime('%H:%M:%S') for t in grids[0].times]
    par = physics.link_params(panel, links); vcut = 0.6 * par['free_speed_kmh']; lanes = np.maximum(par['lanes'], 1.0)
    bid = block_ids(panel, links)
    prof = np.load(G2W / 'g2' / f'{panel}_profile_all.npz', allow_pickle=True)
    tz = np.load(G2V2 / 'rows' / f'{panel}_tmpl.npz', allow_pickle=True)
    assert links == [str(x) for x in prof['links']] == [str(x) for x in tz['links']]
    tmed, tiqr = template_from_hist(tz['hist'].sum(0))
    SS = SplitStats(np.stack([g.value['speed_kmh'] for g in grids]), np.stack([g.value['flow_vph'] for g in grids]),
                    np.stack([g.eligible for g in grids]), vcut, lanes, bid)
    wi = pd.read_csv(R.RELEASE / 'task2' / panel / split / 'window_index.csv')
    return dict(grids=grids, links=links, times0=times0, nT=len(times0), par=par, vcut=vcut, lanes=lanes, bid=bid, prof=prof,
                tmed=tmed, tiqr=tiqr, SS=SS, wi=wi)


def split_feats(d, st, li, cc, a_):
    """Adds split statistics to the row dict `d`.

    Args:
      d: column dict of the rows being built.
      st: dict of (L, T) arrays from SplitStats.loo.
      li: link index per row.
      cc: slot of the cell per row.
      a_: first hole slot; the pre and post columns read slots a_-1 and a_+LG.
    """
    for k, x in st.items(): d[k] = x[li, cc].astype(np.float32)
    for k, c_ in (('pre', a_ - 1), ('post', a_ + LG)):
        d[f's_v_{k}'] = st['s_v'][li, c_].astype(np.float32); d[f's_qf_{k}'] = st['s_qf'][li, c_].astype(np.float32)
        d[f's_bqf_{k}'] = st['s_bqf'][li, c_].astype(np.float32)


def build_official(split, panels):
    """Builds the feature rows of every eligible cell of the official holes.

    gap_min is the time in minutes since the panel's previous official window
    (all dates of the split, in origin order), the selector spacing feature.

    Returns:
      Frame with the build_ph columns except truth, plus timestamp, link_id
      and window_id.
    """
    feats, ms, mf = models(); parts = []
    for panel in panels:
        C = panel_ctx(panel, split); byd = {g.date: (i, g) for i, g in enumerate(C['grids'])}
        wi = C['wi'].copy(); wi['T'] = pd.to_datetime(wi.forecast_origin, utc=True); wi = wi.sort_values('T').reset_index(drop=True)
        wi['gap_min'] = wi['T'].diff().dt.total_seconds() / 60
        for _, w in wi.iterrows():
            di, g = byd[w.date]; a_ = C['times0'].index(w.forecast_start[11:19]); T = a_ - 1; seg = slice(a_, a_ + LG)
            V = g.value['speed_kmh']; Q = g.value['flow_vph']; O = g.value['occupancy']; E = g.eligible
            # The official hole (T+1 .. T+18) is blank in the released layer.
            assert np.isnan(V[:, seg]).all() and np.isnan(Q[:, seg]).all()
            wd = pd.Timestamp(w.date).weekday()
            f = gap_matrix(V, Q, O, E, a_, C['nT'], C['par'], C['vcut'], C['prof'][f'speed_kmh_{wd}'], C['prof'][f'flow_vph_{wd}'], C['bid'],
                           C['tmed'], C['tiqr'], dict(cond=w.condition, gap_min=float(w.gap_min) if np.isfinite(w.gap_min) else np.nan))
            pv, pq = g2v2_predict(ms, mf, feats, f, panel, C['lanes'])
            li, si = np.where(E[:, seg]); st = C['SS'].loo(di); cc = a_ + si
            d = {'date': w.date, 'day': np.int16(int(w.date[8:10])), 'origin': np.int16(T), 'cond': w.condition,
                 'li': li.astype(np.int16), 'step_i': si.astype(np.int8),
                 'ch_v': pv[li, si].astype(np.float32), 'ch_ql': (pq[li, si] / C['lanes'][li]).astype(np.float32)}
            for k in KEEP_F: d['f_' + k] = np.asarray(f[k])[li, si].astype(np.float32)
            split_feats(d, st, li, cc, a_)
            z = pd.DataFrame(d); z.insert(0, 'panel', panel)
            z['timestamp'] = [w.date + 'T' + C['times0'][a_ + s] + 'Z' for s in si]; z['link_id'] = [C['links'][l] for l in li]
            z['window_id'] = w.window_id; parts.append(z)
        print(split, panel, 'official windows', len(wi), flush=True)
        del C
    return pd.concat(parts, ignore_index=True)


def build_pseudo_private(panels):
    """Writes work/ph_private_<panel>.parquet: build_ph.py's validation path on the private month."""
    feats, ms, mf = models()
    PW = B.pseudo_windows_validation(pd.read_parquet(HERE / 'work' / 'cands_private.parquet'), check=False)
    PW.to_parquet(HERE / 'work' / 'windows_private.parquet', index=False)
    CH = parent_state('private')
    t0 = time.time()
    for panel in panels:
        out = HERE / 'work' / f'ph_private_{panel}.parquet'
        if out.exists(): print('skip', out.name, flush=True); continue
        C = panel_ctx(panel, 'private'); grids = C['grids']; nT = C['nT']; nL = len(C['links']); lanes = C['lanes']
        off_holes = {}
        for _, w in C['wi'].iterrows(): off_holes.setdefault(w.date, []).append(C['times0'].index(w.forecast_start[11:19]))
        wins = PW[PW.panel == panel].copy(); wins['origin'] = wins.t
        ch = CH[CH.panel == panel]; chd = {d: x for d, x in ch.assign(date=ch.timestamp.str[:10]).groupby('date')}
        li_ = {l: j for j, l in enumerate(C['links'])}; ti_ = {t: j for j, t in enumerate(C['times0'])}
        parts = []; nw = dict(cand=len(wins), kept=0, off_overlap=0, dup=0, edge=0)
        for di, g in enumerate(grids):
            ws = wins[wins.date == g.date].sort_values(['mult', 'origin'], ascending=[False, True])
            if not len(ws): continue
            oh = off_holes.get(g.date, []); kept = []
            for _, w in ws.iterrows():
                a_ = int(w.origin) + 1
                if a_ - HISTN - 1 < 0 or a_ + LG + 12 > nT: nw['edge'] += 1; continue
                if any((a_ - 13 < o + LG) and (o < a_ + LG + 12) for o in oh): nw['off_overlap'] += 1; continue
                if any(abs(a_ - k) < LG for k, _ in kept): nw['dup'] += 1
                kept.append((a_, w))
            if not kept: continue
            wd = pd.Timestamp(g.date).weekday(); HV, HQ = C['prof'][f'speed_kmh_{wd}'], C['prof'][f'flow_vph_{wd}']
            st = C['SS'].loo(di)
            cz = chd.get(g.date); CV = np.full((nL, nT), np.nan); CQ = np.full((nL, nT), np.nan)
            if cz is not None:
                r_ = cz.link_id.map(li_).to_numpy(); c_ = cz.timestamp.str[11:19].map(ti_).to_numpy()
                CV[r_, c_] = cz.speed_kmh.to_numpy(); CQ[r_, c_] = cz.flow_vph.to_numpy()
            for a_, w in kept:
                T = a_ - 1; hs = slice(T - HISTN, T); seg = slice(a_, a_ + LG)
                V = g.value['speed_kmh'].copy(); Q = g.value['flow_vph'].copy(); O = g.value['occupancy'].copy(); E = g.eligible
                tv = V[:, seg].copy(); tq = Q[:, seg].copy()
                for X, Cm in ((V, CV), (Q, CQ)):
                    X[:, hs] = np.where(np.isnan(X[:, hs]) & np.isfinite(Cm[:, hs]), Cm[:, hs], X[:, hs])
                obs_t = np.isfinite(tv) & np.isfinite(tq)
                for X in (V, Q, O): X[:, seg] = np.nan
                win = dict(cond=w.condition, gap_min=float(w.gap_min) if np.isfinite(w.gap_min) else np.nan)
                ok = E[:, seg] & np.isfinite(tv) & np.isfinite(tq)
                if not ok.any(): continue
                f = gap_matrix(V, Q, O, E, a_, nT, C['par'], C['vcut'], HV, HQ, C['bid'], C['tmed'], C['tiqr'], win)
                pv, pq = g2v2_predict(ms, mf, feats, f, panel, lanes)
                li, si = np.where(ok)
                d = {'date': g.date, 'day': np.int16(int(g.date[8:10])), 'origin': np.int16(T), 'cond': w.condition,
                     'li': li.astype(np.int16), 'step_i': si.astype(np.int8), 'obs': obs_t[li, si],
                     'y_v': tv[li, si].astype(np.float32), 'y_ql': (tq[li, si] / lanes[li]).astype(np.float32),
                     'ch_v': pv[li, si].astype(np.float32), 'ch_ql': (pq[li, si] / lanes[li]).astype(np.float32),
                     'vcut': C['vcut'][li].astype(np.float32), 'bid': C['bid'][li].astype(np.int16)}
                for k in KEEP_F: d['f_' + k] = np.asarray(f[k])[li, si].astype(np.float32)
                split_feats(d, st, li, a_ + si, a_)
                parts.append(pd.DataFrame(d)); nw['kept'] += 1
        D = pd.concat(parts, ignore_index=True); D.insert(0, 'panel', panel)
        D.to_parquet(out, index=False)
        print(panel, 'private windows', nw, 'cells', len(D), round(time.time() - t0, 1), flush=True)
        del C, grids


def main():
    """Runs `pseudo private [panels]` or `official <split> [panels]`."""
    mode = sys.argv[1]
    if mode == 'pseudo':
        assert sys.argv[2] == 'private'
        build_pseudo_private(sys.argv[3:] or PANELS)
    else:
        split = sys.argv[2]; out = HERE / 'work' / f'off_rows_{split}.parquet'
        assert not out.exists(), f'refuse to overwrite {out}'
        Of = build_official(split, sys.argv[3:] or PANELS); Of.to_parquet(out, index=False); print(split, 'official rows', len(Of))


if __name__ == '__main__':
    main()
