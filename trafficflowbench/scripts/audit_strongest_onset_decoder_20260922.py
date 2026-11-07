"""Bounded decoder addendum; all selection uses the three development folds."""

# Role: research audit, kept as a record; the reproduction pipeline does not
# run it, but audit_strongest_onset_selected_20260922.py imports its decoders.
# It keeps the head probabilities that audit_strongest_onset_20260922 saved for
# the three development folds and compares four decoders (RULES):
# reference63_ensemble50 (.5 * reference127 + .5 * capacity63 with its own
# threshold); panel_shrink25 and panel_shrink50 (one threshold per panel, moved
# 25 % or 50 % of the way in log-odds from the fold's global threshold toward
# the panel's best threshold); and expected_jaccard (per window, the
# probability-ranked prefix with the highest expected IoU). The frozen choice
# is written to final_selection.json.

from __future__ import annotations
import itertools,importlib.util,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('onset_explore',ROOT/'scripts/audit_strongest_onset_20260922.py')
X=importlib.util.module_from_spec(sp);sp.loader.exec_module(X)
Q=X.Q;M=X.M;TE=X.TE;OUT=X.OUT
# 64-point Gauss-Legendre rule moved from [-1, 1] to [0, 1], for the integral
# in expected_utilities().
NODES,WEIGHTS=np.polynomial.legendre.leggauss(64)
NODES=(NODES+1)/2;WEIGHTS=WEIGHTS/2
RULES=['reference63_ensemble50','panel_shrink25','panel_shrink50','expected_jaccard']


def expected_utilities(p):
    """Exact integral under independent Bernoulli marginal probabilities.

    E[IoU(S,Y)] = sum(p_i in S) * integral_0^1
       t^(|S|-1) product_{j outside S}(1-p_j+p_j*t) dt.
    Select among every nonempty probability-ranked prefix, no size cap.
    """
    # Uses 1/n = integral_0^1 t^(n-1) dt and E[t^Z] = product(1 - p_j + p_j*t)
    # for the number Z of queued cells outside the prefix. Column k below is the
    # prefix of the k + 1 most probable cells.
    p=np.asarray(p,float);order=np.argsort(-p,kind='stable');r=p[order]
    terms=np.log1p((NODES[:,None]-1)*r[None,:])
    suffix=np.cumsum(terms[:,::-1],axis=1)[:,::-1]
    after=np.c_[suffix[:,1:],np.zeros(len(NODES))]
    power=np.log(NODES[:,None])*np.arange(len(r))[None,:]
    expected_inverse=WEIGHTS@np.exp(after+power)
    utility=np.cumsum(r)*expected_inverse
    return order,utility


def expected_decode(d,p,temperature,odds):
    """Decodes each window as its probability-ranked prefix with the highest expected IoU.

    Probabilities are first rescaled as expit(logit(p) / temperature + log(odds)).

    Returns:
        Boolean decision per row of `d`.
    """
    posterior=expit(logit(np.clip(p,1e-10,1-1e-10))/temperature+np.log(odds))
    pred=np.zeros(len(d),bool)
    for ix in d.groupby('window_id',sort=False).indices.values():
        order,u=expected_utilities(posterior[ix]);k=int(np.argmax(u))+1
        pred[ix[order[:k]]]=True
    return pred


def verify_formula():
    """Checks expected_utilities() against brute-force enumeration on 24 random 6-cell windows.

    The assertions fail when the formula error exceeds 1e-12 or when some
    nonempty subset beats the best ranked prefix.

    Returns:
        Summary dict for the audit record.
    """
    rng=np.random.default_rng(20260922);worst=0.
    for _ in range(24):
        p=rng.uniform(.01,.99,6);order,u=expected_utilities(p)
        truth=np.array(list(itertools.product([0,1],repeat=6)),bool)
        mass=np.prod(np.where(truth,p,1-p),axis=1)
        brute=[]
        for k in range(1,7):
            selected=np.zeros(6,bool);selected[order[:k]]=True
            iou=(truth & selected).sum(axis=1)/(truth | selected).sum(axis=1)
            brute.append(float(mass@iou))
        worst=max(worst,float(np.max(np.abs(np.asarray(brute)-u))))
        all_scores=[]
        for bits in itertools.product([0,1],repeat=6):
            selected=np.array(bits,bool)
            if not selected.any():continue
            all_scores.append(float(mass@((truth & selected).sum(axis=1)/(truth | selected).sum(axis=1))))
        assert abs(max(all_scores)-max(u))<1e-12
    assert worst<1e-12
    return {'small_exact_enumeration_cases':24,'max_absolute_expected_IoU_error':worst,
            'best_ranked_prefix_matches_best_of_all_nonempty_subsets':True,
            'assumption':'Conditional independence is a modeling approximation, not a claim about traffic queues'}


def panel_calibration(ca,p,global_t,alpha):
    """Chooses a shrunk decision threshold per panel on calibration rows.

    For each panel the best GRID threshold is found (ties go to the one
    closest to `global_t`) and then moved from `global_t` toward it in
    log-odds: expit((1 - alpha) * logit(global_t) + alpha * logit(local)).

    Returns:
        (panel -> threshold, panel -> details).
    """
    thresholds={};details={}
    for panel,ix in ca.groupby('panel',sort=False).indices.items():
        g=ca.iloc[ix].reset_index(drop=True);gp=p[ix];rows=[]
        for t in M.GRID:
            score=TE.s_queue(Q.frame(g,Q.decode(g,gp,{'queue_onset':t})))
            rows.append((float(score),float(t)))
        best=max(x[0] for x in rows)
        local=min([t for s,t in rows if abs(s-best)<1e-12],key=lambda t:abs(t-global_t))
        thresholds[panel]=float(expit((1-alpha)*logit(global_t)+alpha*logit(local)))
        details[panel]={'local':local,'shrunk':thresholds[panel],'calibration_windows':g.window_id.nunique()}
    return thresholds,details


def panel_decode(d,p,thresholds):
    """Decodes each panel's rows with that panel's threshold (audit_holdout20_queue.decode)."""
    result=np.zeros(len(d),bool)
    for panel,ix in d.groupby('panel',sort=False).indices.items():
        result[ix]=Q.decode(d.iloc[ix].reset_index(drop=True),p[ix],{'queue_onset':thresholds[panel]})
    return result


def calibrate_decoder(rule,ca,cp,global_t):
    """Fits a decoder's settings on calibration probabilities.

    Args:
        rule: 'panel_shrink25', 'panel_shrink50' or 'expected_jaccard'.
        ca: calibration rows.
        cp: calibration probabilities.
        global_t: the fold's global threshold (reference127).

    Returns:
        Descriptor dict for decode().

    Raises:
        ValueError: unknown rule.
    """
    if rule.startswith('panel_shrink'):
        alpha=.25 if rule=='panel_shrink25' else .5
        thresholds,details=panel_calibration(ca,cp,global_t,alpha)
        return {'rule':rule,'alpha':alpha,'thresholds':thresholds,'panels':details}
    if rule=='expected_jaccard':
        rows=[]
        for temp in [1.,.75,1.25]:
            for odds in [1.,.5,2.]:
                pred=expected_decode(ca,cp,temp,odds)
                score=TE.s_queue(Q.frame(ca,pred))
                rows.append({'temperature':temp,'odds_multiplier':odds,'score':score})
        best=max(rows,key=lambda r:r['score'])
        return {'rule':rule,'selected':best,'calibration_grid':rows}
    raise ValueError(rule)


def decode(rule,d,p,desc):
    """Applies a calibrated decoder descriptor to rows `d`; any other rule uses its single threshold."""
    if rule.startswith('panel_shrink'):return panel_decode(d,p,desc['thresholds'])
    if rule=='expected_jaccard':return expected_decode(d,p,desc['selected']['temperature'],desc['selected']['odds_multiplier'])
    return Q.decode(d,p,{'queue_onset':desc['threshold']})


def main():
    """Scores the four decoders on the three folds and freezes the final selection over all recipes and decoders.

    Raises:
        SystemExit: decoder_addendum.json already exists.
    """
    dest=OUT/'decoder_development';dest.mkdir(parents=True,exist_ok=True)
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':Q.sha(__file__),
      'recipes':RULES,'folds':X.FOLDS,'reference_base':'reference127, same frozen clean development predictions',
      'expected_jaccard':{'nodes':64,'temperature':[1.,.75,1.25],'odds_multiplier':[1.,.5,2.],
         'prefix_sizes':'all 1..physical link count; rank by transformed marginal p',
         'independence_assumption_tested_empirically':'Spatial queues are correlated; formula is a hypothesis and must improve matched forward scoring'},
      'panel_shrink':[.25,.5],'panel_shrink_space':'log-odds threshold toward fold-global threshold',
      'panel_tie_break':'among equally-best local thresholds choose closest to global, avoiding arbitrary extremes',
      'ensemble50':'fixed .5 reference127 + .5 capacity63, separately recalibrated per fold',
      'selection':'same gate: mean delta>0, >=2/3 positive fold deltas, worstfold>=-.015; within .001 prefer simpler decoder',
      'held_scores_not_read_for_addendum_selection':True,'already_seen_research_holdout':True}
    if (OUT/'decoder_addendum.json').exists():raise SystemExit('Refusing to overwrite decoder addendum')
    X.dump(OUT/'decoder_addendum.json',protocol)
    X.dump(dest/'formula_enumeration_test.json',verify_formula())
    # Step 1-5 labels for fullsix().
    old=pd.read_parquet(X.HOLD/'champion_queue_onset_development_features.parquet');early=X.early_truth(old)
    results=[]
    # Decoders reuse the fold probabilities saved by `explore`; no model is refit.
    for fold,*_ in X.FOLDS:
        src=OUT/'development'/fold
        ca=pd.read_parquet(src/'reference127_calibration.parquet')
        ev=pd.read_parquet(src/'reference127_predictions.parquet')
        c63=pd.read_parquet(src/'capacity63_calibration.parquet')
        e63=pd.read_parquet(src/'capacity63_predictions.parquet')
        for a,b in [(ca,c63),(ev,e63)]:pd.testing.assert_frame_equal(a[Q.KEY],b[Q.KEY],check_exact=True)
        global_t=json.loads((src/'reference127_result.json').read_text())['threshold']
        for rule in RULES:
            if rule=='reference63_ensemble50':
                cp=.5*(ca.prob.to_numpy()+c63.prob.to_numpy());ep=.5*(ev.prob.to_numpy()+e63.prob.to_numpy())
                t,grid=Q.calibrate(ca,cp,'queue_onset',M.GRID);desc={'rule':rule,'threshold':t,'calibration_grid':grid}
            else:
                cp=ca.prob.to_numpy();ep=ev.prob.to_numpy();desc=calibrate_decoder(rule,ca,cp,global_t)
            pred=decode(rule,ev,ep,desc);f=X.fullsix(ev,pred,early)
            f.to_parquet(dest/f'{fold}_{rule}_windows.parquet',index=False)
            ev[Q.KEY].assign(prob=ep,pred=pred.astype('int8')).to_parquet(dest/f'{fold}_{rule}_predictions.parquet',index=False)
            r={'fold':fold,'recipe':rule,'S_onset':TE.s_queue(f),'step6_S_onset':TE.s_queue(Q.frame(ev,pred)),
               'calibration':desc,'panel_scores':f.groupby('panel').IoU.mean().to_dict()}
            results.append(r);print('DECODER',fold,rule,r['S_onset'],flush=True)
            X.dump(dest/'results.json',results)
    # Same gate as `explore` over the original recipes and the new decoders;
    # within .001 of the best, the first in `order` wins.
    original=json.loads((OUT/'development_results.json').read_text())
    table=pd.DataFrame([{'fold':r['fold'],'recipe':r['recipe'],'S_onset':r['S_onset']} for r in original+results])
    pivot=table.pivot(index='fold',columns='recipe',values='S_onset').loc[['oct','nov','dec']]
    delta=pivot.subtract(pivot.reference127,axis=0);summary=[]
    for rule in X.RULES+RULES:
        v=delta[rule];eligible=bool(v.mean()>0 and (v>0).sum()>=2 and v.min()>=-.015)
        summary.append({'recipe':rule,'mean_S_onset':float(pivot[rule].mean()),'mean_delta':float(v.mean()),
           'positive_folds':int((v>0).sum()),'worst_fold_delta':float(v.min()),'eligible':eligible,'fold_deltas':v.to_dict()})
    eligible=[r for r in summary if r['eligible']]
    if eligible:
        best=max(r['mean_S_onset'] for r in eligible);short={r['recipe'] for r in eligible if r['mean_S_onset']>=best-.001}
        order=['capacity31','capacity63','no_absolute_clock','no_unconditional_prior','reference127',
               'panel_shrink25','panel_shrink50','reference63_ensemble50','expected_jaccard','legacy_blend25']
        selected=next(r for r in order if r in short)
    else:selected='reference127'
    freeze={'selected_at_utc':datetime.now(timezone.utc).isoformat(),'selected_recipe':selected,
        'model_protocol_sha256':Q.sha(OUT/'protocol.json'),'decoder_addendum_sha256':Q.sha(OUT/'decoder_addendum.json'),
        'summary':summary,'only_first218_development_dates_used_for_new_selection':True,
        'held_before_this_selection':'Previously seen in earlier research; none used in this new recipe-selection computation'}
    X.dump(OUT/'final_selection.json',freeze);table.to_csv(dest/'all_development_scores.csv',index=False)
    print('FINAL_SELECTION',json.dumps(freeze,indent=2),flush=True)


if __name__=='__main__':main()
