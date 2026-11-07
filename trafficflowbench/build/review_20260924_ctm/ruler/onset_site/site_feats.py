"""Per (event/window, candidate site) features at origin T, for train events and pseudo-val onset windows.
Site bottleneck links (bn) = links hitting q/c~1 at breakdown (from bneck.py)."""
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
FV = Path(__file__).resolve().parent.parent / 'fakeval'
sys.path.insert(0, str(HERE))
from pv_sites import SITES, site_of
BN = {
    'D12_I5_N': {'S1': [106, 108, 109], 'S2': [241], 'S3': [15]},
    'D12_I5_S': {'S1': [42], 'S2': [11], 'S3': [92], 'S4': [40]},
    'D7_I10_E': {'S1': [3], 'S2': [35]},
    'D7_I10_W': {'S1': [30]},
    'D7_I210_E': {'S1': [5]},
    'D7_I210_W': {'S1': [32], 'S2': [5], 'S3': [44]},
    'D7_I405_N': {'S1': [350], 'S2': [66]},
    'D7_I405_S': {'S1': [34], 'S2': [19]},
}


def feats(Qd, Vd, Od, cap, vf, T, panel):
    """Qd,Vd: [L, 288] one day. returns dict site->features"""
    out = {}
    for s, bl in BN[panel].items():
        lo, hi = SITES[panel][s]
        bl = np.array(bl)
        r = Qd[bl, :] / cap[bl, None]
        rm = np.nanmean(r, 0)
        rT = rm[T]
        rT1 = np.nanmean(rm[T - 1:T + 1])
        r6 = np.nanmean(rm[T - 5:T + 1])
        r3 = np.nanmean(rm[T - 2:T + 1])
        r6p = np.nanmean(rm[T - 11:T - 5])
        rmx3 = np.nanmax(rm[T - 2:T + 1])
        r12 = np.nanmean(rm[T - 11:T + 1])
        reg = np.arange(max(0, lo - 1), min(len(cap), hi + 2))
        vm = np.nanmin(np.nanmean(Vd[reg, T - 2:T + 1] / vf[reg, None], 1))
        vsd = np.nanmean(np.nanstd(Vd[reg, T - 11:T + 1] / vf[reg, None], 1))
        om = np.nanmax(np.nanmean(Od[reg, T - 2:T + 1], 1)) if Od is not None else np.nan
        out[s] = dict(rT=rT, rT1=rT1, r6=r6, r3=r3, r12=r12, r6p=r6p, rmx3=rmx3, dr=r6 - r6p, vmin=vm, vsd=vsd, occ=om)
    return out


def train_rows():
    ev = pd.read_parquet(HERE / 'events_train.parquet')
    rows = []
    for panel in SITES:
        z = np.load(HERE / 'cache' / f'{panel}_train.npz', allow_pickle=True)
        links = z['links'].astype(str)
        lk = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str}).set_index('link_id').reindex(links)
        cap = lk.capacity_vph.to_numpy(); vf = lk.free_speed_kmh.to_numpy()
        Qa = z['Q']; Va = z['V']; Oa = z['O']; dates = z['dates']
        g = ev[ev.panel == panel]
        for (di, tq), gg in g.groupby(['di', 'tq']):
            T = tq - 6
            if T < 12: continue
            true = set()
            for r in gg.itertuples():
                for i in range(r.a, r.b + 1): true.add(site_of(panel, i))
            Qd = Qa[di].astype(np.float32); Vd = Va[di].astype(np.float32); Od = Oa[di].astype(np.float32)
            f = feats(Qd, Vd, Od, cap, vf, T, panel)
            for s, d in f.items():
                rows.append(dict(split='train', panel=panel, date=str(dates[di]), key=f'{panel}|{dates[di]}|{tq}', T=T, site=s, y=int(s in true), **d))
    return pd.DataFrame(rows)


def pv_rows():
    w = pd.read_parquet(FV / 'windows_validation.parquet'); w = w[w.condition == 'queue_onset']
    ps = pd.read_parquet(HERE / 'pv_sites.parquet').set_index('window_id')
    rows = []
    for panel in SITES:
        links = None
        lk = None
        for r in w[w.panel == panel].itertuples():
            z = np.load(FV / 'days' / f'{panel}_{r.date}.npz', allow_pickle=True)
            day = pd.read_parquet(FV / 'days' / f'{panel}_{r.date}.parquet')
            lnk = z['links'].astype(str); li = {k: i for i, k in enumerate(lnk)}
            times = list(z['times'].astype(str)); ti = {k: i for i, k in enumerate(times)}
            if lk is None:
                lk = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str}).set_index('link_id').reindex(lnk)
                cap = lk.capacity_vph.to_numpy(); vf = lk.free_speed_kmh.to_numpy()
            n = (len(lnk), len(times))
            rr = day.link_id.astype(str).map(li).to_numpy(); cc = day.timestamp.map(ti).to_numpy()
            Qd = np.full(n, np.nan); Qd[rr, cc] = day.flow_vph.to_numpy()
            Vd = np.full(n, np.nan); Vd[rr, cc] = day.speed_kmh.to_numpy()
            Od = np.full(n, np.nan); Od[rr, cc] = day.occupancy.to_numpy()
            Fd = np.zeros(n, bool); Fd[rr, cc] = day.filled.to_numpy().astype(bool)
            # legality: blank everything after T
            T = int(r.t) - 0  # window origin index t: history t-12..t-1, horizon t+1..t+6 (origin time = times[t])
            Qd[:, T + 1:] = np.nan; Vd[:, T + 1:] = np.nan; Od[:, T + 1:] = np.nan
            # at T itself only the masked layer is released: drop Task-1-filled cells there
            for A in (Qd, Vd, Od): A[Fd[:, T], T] = np.nan
            f = feats(Qd, Vd, Od, cap, vf, T, panel)
            ysite = ps.loc[r.window_id, 'ysite'] if r.window_id in ps.index else ''
            ytrue = set(ysite.split('+')) if isinstance(ysite, str) else set()
            for s, d in f.items():
                rows.append(dict(split='pv', panel=panel, date=r.date, key=r.window_id, T=T, site=s, y=int(s in ytrue), **d))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    tr = train_rows(); tr.to_parquet(HERE / 'site_train.parquet', index=False)
    pv = pv_rows(); pv.to_parquet(HERE / 'site_pv.parquet', index=False)
    print(tr.shape, pv.shape)
