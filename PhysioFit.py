#!/usr/bin/env python3
"""EyeSeeCam PLR analysis of binocular pupil covariance CSV files.

The only recording-level exclusion is >=25% reconstruction across all samples.
QC phase flags are descriptive; insufficiently supported individual metrics stay NaN.

Requires numpy, pandas, scipy, and matplotlib. Tkinter is used for the optional GUI.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
GLOBAL_RECONSTRUCTION_EXCLUSION_PERCENT = 25.0
INPUT_COLUMNS = (
    'Time', 'visual_pos3', 'TargetPosHor',
    'LeftPupilCovCol', 'LeftPupilCovColRow', 'LeftPupilCovRow',
    'RightPupilCovCol', 'RightPupilCovColRow', 'RightPupilCovRow',
)

@dataclass(frozen=True)
class Phases:
    light_start: float
    light_end: float
    recording_start: float
    recording_end: float
    fs: float
    light_duration_s: float
    recovery_duration_s: float

def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    x = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, x, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))

def max_run_seconds(mask: np.ndarray, fs: float) -> float:
    return max(((b - a) / fs for a, b in runs(mask))) if np.any(mask) else 0.0

def reconstruction_qc(reconstructed: np.ndarray) -> tuple[float, str]:
    """The only recording-level exclusion: >=25% of all time samples unsupported."""
    reconstructed_percent = float(100 * np.mean(reconstructed))
    status = ('FAIL' if reconstructed_percent >= GLOBAL_RECONSTRUCTION_EXCLUSION_PERCENT
              else 'PASS')
    return reconstructed_percent, status


def determine_phases(df: pd.DataFrame) -> Phases:
    t = df['Time'].to_numpy(float)
    v = df['visual_pos3'].to_numpy(float)
    marker = df['TargetPosHor'].to_numpy(float)
    if len(t) < 1200 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Ungueltige oder zu kurze Zeitachse')
    dt = float(np.median(np.diff(t)))
    fs = 1 / dt
    if not 15 <= fs <= 2000:
        raise ValueError(f'Unplausible Samplingrate: {fs:.1f} Hz')
    if np.any(np.diff(t) > max(0.03, 5 * dt)):
        raise ValueError('Grosse Zeitstempel-Luecken: bitte Rohdatei pruefen')
    marks = {}
    for state in (1, 2, 3):
        rr = runs(v == state)
        if len(rr) != 1:
            raise ValueError(f'Phasencode visual_pos3={state} nicht eindeutig')
        marks[state] = rr[0]
    b0, b1 = marks[1]
    s0, s1 = marks[2]
    r0, r1 = marks[3]
    if not (b1 == s0 and s1 == r0 and (r1 == len(t))):
        raise ValueError('Baseline/Licht/Nachphase sind nicht lueckenlos aufeinanderfolgend')
    if np.nanmedian(marker[s0:s1]) != 1 or np.nanmedian(marker[b0:b1]) != 0 or np.nanmedian(marker[r0:r1]) != 0:
        raise ValueError('TargetPosHor bestaetigt Phasencodes nicht')
    start = float(t[s0])
    end = float(t[s1])
    if start - t[b0] < 2.0:
        raise ValueError('Weniger als 2 s Vor-Baseline')
    return Phases(start, end, float(t[0]), float(t[-1]), fs, end - start, float(t[-1] - end))

def get_area_proxy(df: pd.DataFrame, side: str) -> np.ndarray:
    xx = df[f'{side}PupilCovCol'].to_numpy(float)
    yy = df[f'{side}PupilCovRow'].to_numpy(float)
    xy = df[f'{side}PupilCovColRow'].to_numpy(float)
    det = xx * yy - xy * xy
    with np.errstate(invalid='ignore', over='ignore'):
        area = np.sqrt(det)
    area[~np.isfinite(area) | (xx <= 0) | (yy <= 0) | (det <= 0)] = np.nan
    return area

def phase_windows(ph: Phases) -> dict[str, tuple[float, float]]:
    return {'baseline_pre_2s': (ph.light_start - 2, ph.light_start), 'light': (ph.light_start, ph.light_end), 'recovery_0_10s': (ph.light_end, ph.light_end + 10), 'recovery_10_20s': (ph.light_end + 10, ph.light_end + 20), 'recovery_20_30s': (ph.light_end + 20, ph.light_end + 30)}

def window_mask(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return (t >= start) & (t < end)

def window_qc(t: np.ndarray, orig: np.ndarray, interpolated: np.ndarray,
              smoothed: np.ndarray, start: float, end: float, fs: float) -> dict:
    """Report window coverage; phase-level warnings cannot exclude a recording."""
    if start < t[0] - 0.5 / fs or end > t[-1] + 1.5 / fs:
        return {'status': 'NOT_RECORDED', 'original_fraction': None, 'interpolated_fraction': None, 'remaining_nan_s': None}
    w = window_mask(t, start, end)
    if not w.any():
        return {'status': 'NOT_RECORDED', 'original_fraction': None, 'interpolated_fraction': None, 'remaining_nan_s': None}
    return {'status': 'RECORDED', 'original_fraction': round(float(np.isfinite(orig[w]).mean()), 4), 'interpolated_fraction': round(float(interpolated[w].mean()), 4), 'remaining_nan_s': round(max_run_seconds(~np.isfinite(smoothed[w]), fs), 4)}

def integral_exact(t: np.ndarray, y: np.ndarray, start: float, end: float) -> float:
    tol = 1.5 * float(np.median(np.diff(t)))
    if start < t[0] - tol or end > t[-1] + tol or end <= start:
        return float('nan')
    inner = (t > start) & (t < end)
    sub_t = np.r_[start, t[inner], end]
    sub_y = np.interp(sub_t, t, y)
    return float(trapezoid(sub_y, sub_t))

def features(t: np.ndarray, signal: np.ndarray, ph: Phases, qc: dict[str, dict]) -> dict:
    """Compute AUC and constriction metrics from the displayed analysis signal."""
    out: dict[str, object] = {}
    base_ok = qc['baseline_pre_2s']['status'] != 'NOT_RECORDED'
    light_ok = qc['light']['status'] != 'NOT_RECORDED'
    basemask = window_mask(t, ph.light_start - 2, ph.light_start)
    base = float(np.mean(signal[basemask])) if base_ok and np.isfinite(signal[basemask]).all() else float('nan')
    if not np.isfinite(base) or base <= 0:
        base = float('nan')
    out['Baseline_mean_area_proxy_2s'] = base
    post8 = window_mask(t, ph.light_end + 8, ph.light_end + 10)
    out['Post_baseline_8_10s_after_light_off'] = float(np.mean(signal[post8])) if qc['recovery_0_10s']['status'] != 'NOT_RECORDED' and post8.any() else float('nan')
    post28 = window_mask(t, ph.light_end + 28, ph.light_end + 30)
    out['Post_baseline_28_30s_after_light_off'] = float(np.mean(signal[post28])) if qc['recovery_20_30s']['status'] != 'NOT_RECORDED' and post28.any() else float('nan')
    out['Constriction_minimum_area_proxy'] = float('nan')
    out['Constriction_minimum_normalized'] = float('nan')
    out['Time_to_minimum_s'] = float('nan')
    out['Maximum_Constriction_Velocity_proxy_per_s'] = float('nan')
    out['Maximum_Constriction_Velocity_time_s'] = float('nan')
    minimum = float('nan')
    if light_ok and np.isfinite(base):
        win = window_mask(t, ph.light_start, ph.light_end)
        if win.sum() >= 4 and np.isfinite(signal[win]).all():
            idxs = np.flatnonzero(win)
            mini = int(idxs[np.argmin(signal[idxs])])
            minimum = float(signal[mini])
            out['Constriction_minimum_area_proxy'] = minimum
            out['Constriction_minimum_normalized'] = minimum / base
            out['Time_to_minimum_s'] = float(t[mini] - ph.light_start)
            vel = np.gradient(signal, t)
            if np.isfinite(vel[idxs]).all():
                j = int(idxs[np.argmin(vel[idxs])])
                out['Maximum_Constriction_Velocity_proxy_per_s'] = float(vel[j])
                out['Maximum_Constriction_Velocity_time_s'] = float(t[j] - ph.light_start)
    windows = {'0_10': (0, 10), '10_20': (10, 20), '20_30': (20, 30), '0_20': (0, 20), '0_30': (0, 30)}
    for name, (a, b) in windows.items():
        out[f'Redilation_AUC_{name}_s'] = float('nan')
        out[f'Redilation_AUC_from_minimum_{name}_s'] = float('nan')
        out[f'Redilation_AUC_fractional_recovery_{name}_s'] = float('nan')
        parts = [qc['recovery_0_10s']]
        if b > 10:
            parts.append(qc['recovery_10_20s'])
        if b > 20:
            parts.append(qc['recovery_20_30s'])
        if a >= 10:
            parts = parts[1:]
        if a >= 20:
            parts = parts[1:]
        if not (np.isfinite(base) and all((q['status'] != 'NOT_RECORDED' for q in parts))):
            continue
        lo, hi = (ph.light_end + a, ph.light_end + b)
        segment = (t >= lo - 0.03) & (t <= hi + 0.03)
        if lo < t[0] or hi > t[-1] + 1.5 * np.median(np.diff(t)) or (not np.isfinite(signal[segment]).all()):
            continue
        auc = integral_exact(t, 1 - signal / base, lo, hi)
        out[f'Redilation_AUC_{name}_s'] = auc
        if np.isfinite(minimum):
            out[f'Redilation_AUC_from_minimum_{name}_s'] = integral_exact(t, (signal - minimum) / base, lo, hi)
            amplitude = base - minimum
            if amplitude > 0.05 * base:
                out[f'Redilation_AUC_fractional_recovery_{name}_s'] = integral_exact(t, (signal - minimum) / amplitude, lo, hi)
    a0 = out['Redilation_AUC_0_10_s']
    a1 = out['Redilation_AUC_10_20_s']
    a20 = out['Redilation_AUC_0_20_s']
    if all((np.isfinite(x) for x in [a0, a1, a20])) and (not np.isclose(a0 + a1, a20, atol=1e-06)):
        raise AssertionError('AUC(0-20) != AUC(0-10) + AUC(10-20)')
    return out

def subject_label(path: Path) -> str:
    m = re.search('P_\\d+', path.name, flags=re.IGNORECASE)
    return m.group(0) if m else re.sub('[^A-Za-z0-9_.-]', '_', path.stem)[:70]

def write_final_analysis_figure(path: Path, t: np.ndarray, ph: Phases, final: np.ndarray, feat: dict, qc: dict, label: str) -> None:
    """Plot the final area proxy and baseline-normalized AUC windows."""
    x = t - ph.light_start
    off = ph.light_duration_s
    base = feat['Baseline_mean_area_proxy_2s']
    norm = final / base if np.isfinite(base) and base > 0 else np.full(len(final), np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [1, 1.08]}, constrained_layout=True)
    for ax in axes:
        ax.axvspan(0, off, color='#edbd65', alpha=0.26, label='Lichtstimulus')
        ax.axvline(0, color='#b47c1f', lw=0.9, ls='--')
        ax.axvline(off, color='#b47c1f', lw=0.9, ls='--')
        ax.grid(alpha=0.16)
    axes[0].plot(x, final, color='#276295', lw=1.55, label='Bereinigtes und geglaettetes Analysesignal')
    axes[0].set_ylabel('Pupillenflaeche (Proxy)')
    axes[0].legend(loc='upper right', fontsize=9)
    for lo, hi, color, lab in [(0, 10, '#79accb', 'AUC 0-10 s'), (10, 20, '#79bda8', 'AUC 10-20 s'), (20, 30, '#ad9bd6', 'AUC 20-30 s')]:
        if off + lo <= x[-1] + 0.02:
            axes[1].axvspan(off + lo, min(off + hi, x[-1]), color=color, alpha=0.2, label=lab)
    axes[1].axhline(1, color='#777', ls=':', lw=1.2, label='Baseline = 1')
    axes[1].plot(x, norm, color='#206b59', lw=1.55, label='Geglaettet und baseline-normalisiert')
    axes[1].set_ylabel('Pupillenflaeche / Baseline')
    axes[1].set_xlabel('Zeit relativ zu Lichtbeginn (s)')
    axes[1].legend(loc='upper right', fontsize=8, ncol=3)

    def fmt(v):
        try:
            return f'{float(v):.3f}' if np.isfinite(float(v)) else 'n.v.'
        except (TypeError, ValueError):
            return 'n.v.'
    summary = 'AUC 0-10: ' + fmt(feat['Redilation_AUC_0_10_s']) + '    |    AUC 10-20: ' + fmt(feat['Redilation_AUC_10_20_s']) + '    |    AUC 0-20: ' + fmt(feat['Redilation_AUC_0_20_s']) + '\nAUC 20-30: ' + fmt(feat['Redilation_AUC_20_30_s']) + '    |    AUC 0-30: ' + fmt(feat['Redilation_AUC_0_30_s']) + '    |    AUC vom Minimum 0-10: ' + fmt(feat['Redilation_AUC_from_minimum_0_10_s'])
    axes[1].text(0.01, 0.02, summary, transform=axes[1].transAxes, fontsize=8, ha='left', va='bottom', bbox=dict(boxstyle='round,pad=.40', facecolor='white', alpha=0.87, edgecolor='#cccccc'))
    end = min(x[-1], off + 30)
    axes[1].set_xlim(min(x[0], -2), end)
    fig.suptitle(f'Final analysierte PLR-Kurve – {label}\nBeide Augen separat bereinigt und kombiniert; unten: vollstaendige AUC-Analyse', fontsize=12)
    fig.savefig(path, dpi=165)
    plt.close(fig)

def reconstruct_eye(raw: np.ndarray, t: np.ndarray, ph: Phases, bin_duration: float=0.2) -> dict:
    """Find stable 200-ms measurement bins and reconstruct the per-eye trace."""
    n = len(raw)
    bin_idx = np.floor((t - t[0]) / bin_duration).astype(int)
    nb = int(bin_idx[-1]) + 1
    centers_t = t[0] + (np.arange(nb) + 0.5) * bin_duration
    pre = (t >= ph.light_start - 2) & (t < ph.light_start)
    raw_base = float(np.nanmedian(raw[pre])) if np.isfinite(raw[pre]).any() else float('nan')
    broad = (t >= t[0]) & (t < ph.light_start)
    global_ref = float(np.nanmedian(raw[broad])) if np.isfinite(raw[broad]).any() else float('nan')
    if not np.isfinite(global_ref) or global_ref <= 0:
        vals = raw[np.isfinite(raw) & (raw > 0)]
        global_ref = float(np.median(vals)) if len(vals) else float('nan')
    if not np.isfinite(global_ref) or global_ref <= 0:
        nb = int(bin_idx[-1]) + 1
        return {'usable': False, 'raw': raw, 'trace': np.full(n, np.nan), 'anchor_bin_fraction': 0.0, 'valid': np.zeros(n, bool), 'baseline': float('nan'), 'anchors': np.full(nb, np.nan), 'bin_time': t[0] + (np.arange(nb) + 0.5) * bin_duration, 'bin_quality': np.zeros(nb, bool), 'bin_occupancy': np.zeros(nb), 'reconstruction_fraction': 1.0, 'longest_gap_s': ph.recording_end - ph.recording_start}
    cluster_width = max(0.09 * global_ref, 0.7)
    centers = np.full(nb, np.nan)
    occupancy = np.zeros(nb)
    for i in range(nb):
        v = raw[bin_idx == i]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) < 7:
            continue
        v = np.sort(v)
        end_idx = np.searchsorted(v, v + cluster_width, side='right')
        counts = end_idx - np.arange(len(v))
        j = int(np.argmax(counts))
        k = int(end_idx[j])
        centers[i] = float(np.median(v[j:k]))
        occupancy[i] = counts[j] / len(v)
    accepted = np.isfinite(centers) & (occupancy >= 0.56)
    if accepted.sum() < 8:
        return {'usable': False, 'raw': raw, 'trace': np.full(n, np.nan), 'anchor_bin_fraction': 0.0, 'valid': np.zeros(n, bool), 'baseline': float('nan'), 'anchors': centers, 'bin_time': centers_t, 'bin_quality': accepted, 'bin_occupancy': occupancy, 'reconstruction_fraction': 1.0, 'longest_gap_s': ph.recording_end - ph.recording_start}
    provisional = np.interp(centers_t, centers_t[accepted], centers[accepted])
    med = median_filter(provisional, size=17, mode='nearest')
    near_light = (abs(centers_t - ph.light_start) < 1.1) | (abs(centers_t - ph.light_end) < 1.1)
    implausible = (abs(centers - med) > np.maximum(0.31 * global_ref, 0.4 * med)) & ~near_light
    accepted &= ~implausible
    if accepted.sum() < 8:
        return {'usable': False, 'raw': raw, 'trace': np.full(n, np.nan), 'anchor_bin_fraction': 0.0, 'valid': np.zeros(n, bool), 'baseline': float('nan'), 'anchors': centers, 'bin_time': centers_t, 'bin_quality': accepted, 'bin_occupancy': occupancy, 'reconstruction_fraction': 1.0, 'longest_gap_s': ph.recording_end - ph.recording_start}
    interp = np.interp(centers_t, centers_t[accepted], centers[accepted])
    interp = gaussian_filter1d(interp, sigma=1.65, mode='nearest')
    full = np.interp(t, centers_t, interp)
    bin_good = accepted[bin_idx]
    interp_mask = ~bin_good
    basewin = (centers_t >= ph.light_start - 2) & (centers_t < ph.light_start)
    base = float(np.mean(interp[basewin])) if basewin.sum() else float('nan')
    run_max = max_run_seconds(~accepted, 1 / bin_duration)
    return {'usable': np.isfinite(base) and base > 0, 'raw': raw, 'trace': full, 'anchor_bin_fraction': float(accepted.mean()), 'valid': bin_good, 'baseline': base, 'anchors': centers, 'bin_time': centers_t, 'bin_quality': accepted, 'bin_occupancy': occupancy, 'reconstruction_fraction': float(interp_mask.mean()), 'longest_gap_s': float(run_max), 'interpolated_mask': interp_mask, 'baseline_original_fraction': float(accepted[basewin].mean()) if basewin.any() else 0.0}

def _potential_tracking_events(x: np.ndarray, t: np.ndarray, ph: Phases, baseline: float) -> np.ndarray:
    """Flag abrupt changes outside stimulus transitions without modifying values."""
    mask = np.zeros(len(t), bool)
    if not np.isfinite(baseline):
        return mask
    dt = float(np.median(np.diff(t)))
    offset = max(1, int(round(0.65 / dt)))
    diff = np.abs(x[offset:] - x[:-offset])
    bad = diff > 0.27 * baseline
    idx = np.flatnonzero(bad) + offset
    near = (np.abs(t[idx] - ph.light_start) < 1.3) | (np.abs(t[idx] - ph.light_end) < 1.3)
    idx = idx[~near]
    mask[idx] = True
    return mask

def add_supported_fine_structure(pred: np.ndarray, track: dict, t: np.ndarray, ph: Phases, baseline: float) -> np.ndarray:
    """Add only low-frequency residual structure supported by accepted anchors."""
    tb = np.asarray(track['bin_time'], dtype=float)
    yb = np.asarray(track['anchors'], dtype=float)
    base_good = np.asarray(track['bin_quality'], bool) & (np.asarray(track['bin_occupancy'], float) >= 0.72) & np.isfinite(yb) & (yb > 0)
    expect = np.interp(tb, t, pred)
    deviation = (yb - expect) / baseline
    base_good &= np.isfinite(deviation) & (np.abs(deviation) < 0.115)
    if base_good.sum() < 9:
        return pred
    delta = np.zeros(len(t), float)
    for aa, bb in [(float(t[0]), ph.light_start), (ph.light_start, ph.light_end), (ph.light_end, float(t[-1]) + 1e-05)]:
        phase = (tb >= aa) & (tb < bb)
        valid = base_good & phase
        if valid.sum() < 6:
            continue
        tid = tb[phase]
        val = deviation[phase]
        good = valid[phase]
        a = np.full(len(tid), np.nan)
        a[good] = val[good]
        clean = np.zeros(len(tid), float)
        weight = np.zeros(len(tid), float)
        for i in range(len(tid)):
            d = np.abs(tid - tid[i])
            in_win = good & (d <= 0.85)
            if in_win.sum() < 4:
                continue
            z = a[in_win]
            med = float(np.median(z))
            mad = float(np.median(np.abs(z - med)))
            if mad > 0.035:
                continue
            if np.abs(med) > 0.08:
                continue
            clean[i] = med
            weight[i] = 1.0
        if weight.sum() < 3:
            continue
        sm = gaussian_filter1d(clean * weight, sigma=4.0, mode='constant')
        sw = gaussian_filter1d(weight, sigma=4.0, mode='constant')
        residual = sm / (sw + 0.12)
        residual *= np.clip(sw / 0.65, 0, 1) ** 2
        residual = gaussian_filter1d(residual, sigma=5.0, mode='nearest')
        residual = np.clip(residual, -0.03, 0.03)
        idx = (t >= aa) & (t < bb)
        if not idx.any():
            continue
        times = t[idx]
        r = np.interp(times, tid, residual, left=0, right=0)
        edge = np.minimum((times - aa) / 0.55, (bb - times) / 0.55)
        r *= np.clip(edge, 0, 1)
        delta[idx] = r * baseline
    fine = pred + delta
    return np.maximum(fine, 0.0001)

def fit_eye_physiology(track: dict, t: np.ndarray, ph: Phases) -> dict:
    """Fit a per-eye baseline, constriction, and recovery model from accepted bins."""
    empty = dict(
        usable=False,
        model=np.full_like(t, np.nan, dtype=float),
        baseline=np.nan,
        light_depth=np.nan,
        recovery_tau_s=np.nan,
        recovery_plateau=np.nan,
        recovery_delay_s=np.nan,
        median_relative_residual=np.nan,
        fraction_fitted=np.nan,
        bin_residual=np.nan,
        baseline_support=0,
        light_support=0,
        recovery_support=0,
        full_support=0,
        fit_uncertain=True,
    )
    if not track['usable']:
        return empty
    tb = track['bin_time']
    yb = track['anchors']
    good = track['bin_quality'] & np.isfinite(yb) & (yb > 0)
    baseline_win = (tb >= ph.light_start - 2) & (tb < ph.light_start)
    during_win = (tb >= ph.light_start + 0.04) & (tb < ph.light_end)
    recover_win = (tb >= ph.light_end) & (tb < min(ph.light_end + 30, t[-1]))
    nb = int((good & baseline_win).sum())
    nl = int((good & during_win).sum())
    nr = int((good & recover_win).sum())
    extended_win = (tb >= ph.light_start - 7) & (tb < ph.light_start - 2)
    ne = int((good & extended_win).sum())
    if nb + ne < 3 or nl < 3 or nr < 3:
        return {**empty, 'baseline_support': nb, 'light_support': nl, 'recovery_support': nr}
    ref_recent = float(np.median(yb[good & baseline_win])) if nb else float('nan')
    ref_early = float(np.median(yb[good & extended_win])) if ne else float('nan')
    baseline_suspect = nb < 7 or (ne >= 3 and np.isfinite(ref_recent) and (abs(ref_recent - ref_early) > 0.18 * ref_early))
    baseline = ref_early if baseline_suspect and ne >= 3 else ref_recent
    if not np.isfinite(baseline) or baseline <= 0:
        return empty
    t_light = tb[good & during_win] - ph.light_start
    x_light = yb[good & during_win] / baseline
    occ = track['bin_occupancy'][good & during_win]
    lat = 0.18
    duration = ph.light_duration_s

    def constriction(xx, depth, tau, escape):
        z = np.maximum(xx - lat, 0)
        den = max(1e-06, 1 - np.exp(-max(duration - lat, 0.02) / tau))
        escape_start = min(2.0, 0.5 * duration)
        drift = np.maximum(xx - escape_start, 0) / max(duration - escape_start, 0.1)
        return 1 - depth * (1 - np.exp(-z / tau)) / den + escape * drift

    def objective_c(params):
        d, tau, escape = params
        pred = constriction(t_light, d, tau, escape)
        reg = np.array([0.02 * (tau - 0.9), 0.16 * escape])
        return np.r_[np.sqrt(np.maximum(0.25, occ)) * (pred - x_light), reg]
    deep_guess = float(np.clip(1 - np.quantile(x_light, 0.3), 0.12, 0.86))
    cfit = least_squares(objective_c, x0=[deep_guess, 0.8, 0.02], bounds=([0.02, 0.45, 0], [0.94, 3.0, 0.2]), loss='soft_l1', f_scale=0.1, max_nfev=300)
    depth, tauc, escape = cfit.x
    end_level = 1 - depth + escape
    u = tb[good & recover_win] - ph.light_end
    yy = yb[good & recover_win] / baseline
    ww = np.sqrt(np.maximum(0.25, track['bin_occupancy'][good & recover_win]))

    def recovery(ss, plateau, tau, delay):
        s = np.maximum(ss - delay, 0.0)
        return end_level + (plateau - end_level) * -np.expm1(-s / tau)

    def objective_r(params):
        plateau, tau, delay = params
        pred = recovery(u, plateau, tau, delay)
        reg = np.array([0.025 * (plateau - 1.0), 0.005 * (tau - 6.0), 0.02 * delay])
        return np.r_[ww * (pred - yy), reg]
    p_guess = np.clip(np.median(yy[u >= max(0, np.max(u) - 5)]) if np.any(u >= np.max(u) - 5) else np.median(yy), max(0.65, end_level + 0.06), 1.1)
    rfit = least_squares(objective_r, x0=[max(p_guess, end_level + 0.08), 5.0, 0.2], bounds=([end_level + 0.01, 0.6, 0.0], [max(1.15, end_level + 0.05), 25.0, 2.6]), loss='soft_l1', f_scale=0.12, max_nfev=400)
    plateau, taur, delay = rfit.x
    pred = np.full(len(t), baseline, dtype=float)
    in_light = (t >= ph.light_start) & (t < ph.light_end)
    after = t >= ph.light_end
    pred[in_light] = baseline * constriction(t[in_light] - ph.light_start, depth, tauc, escape)
    pred[after] = baseline * recovery(t[after] - ph.light_end, plateau, taur, delay)
    pred = add_supported_fine_structure(pred, track, t, ph, baseline)
    m_good = good & (baseline_win | during_win | recover_win)
    if m_good.any():
        r = np.abs((yb[m_good] - np.interp(tb[m_good], t, pred)) / baseline)
        median_err = float(np.median(r))
        fitted_fraction = float(np.mean(r < 0.17))
    else:
        median_err = float('inf')
        fitted_fraction = 0.0
    nbin_base = int(baseline_win.sum())
    nbin_light = int(during_win.sum())
    nbin_rec = int(recover_win.sum())
    frac_base = nb / max(nbin_base, 1)
    frac_light = nl / max(nbin_light, 1)
    frac_rec = nr / max(nbin_rec, 1)
    uncertain = baseline_suspect or median_err > 0.13 or fitted_fraction < 0.72 or (frac_base < 0.7) or (frac_light < 0.55) or (frac_rec < 0.65) or (track['longest_gap_s'] > 1.0)
    return dict(
        usable=True,
        model=pred,
        baseline=baseline,
        light_depth=float(depth),
        light_tau_s=float(tauc),
        pupillary_escape_fraction=float(escape),
        recovery_tau_s=float(taur),
        recovery_plateau=float(plateau),
        recovery_delay_s=float(delay),
        median_relative_residual=median_err,
        fraction_fitted=fitted_fraction,
        baseline_support=nb,
        light_support=nl,
        recovery_support=nr,
        baseline_anchor_fraction=frac_base,
        light_anchor_fraction=frac_light,
        recovery_anchor_fraction=frac_rec,
        longest_gap_s=track['longest_gap_s'],
        baseline_suspect=bool(baseline_suspect),
        baseline_from_earlier_window=bool(baseline_suspect and ne >= 3),
        fit_uncertain=bool(uncertain),
    )

def combine_physiology_models(left: dict, right: dict, leftfit: dict, rightfit: dict, t: np.ndarray, ph: Phases) -> dict:
    fitted = {k: (a, b) for k, a, b in [('Left', left, leftfit), ('Right', right, rightfit)] if b['usable']}
    if not fitted:
        raise ValueError('Kein Auge hat genug stabile Messpunkte in Baseline, Licht und Nachphase')
    quality = {k: 1.5 * b['median_relative_residual'] + 0.5 * (1 - b['baseline_anchor_fraction']) + 0.3 * (1 - b['light_anchor_fraction']) + 0.3 * (1 - b['recovery_anchor_fraction']) + 0.15 * min(b['longest_gap_s'] / 3, 1) for k, (a, b) in fitted.items()}
    best = min(quality, key=quality.get)
    decision = best
    discrepancy = np.zeros(len(t), bool)
    if len(fitted) == 2:
        l = leftfit['model'] / leftfit['baseline']
        r = rightfit['model'] / rightfit['baseline']
        discrepancy = np.abs(l - r) > 0.15
        if discrepancy.mean() < 0.04 and abs(quality['Left'] - quality['Right']) < 0.12:
            decision = 'both'
    selected = ['Left', 'Right'] if decision == 'both' else [decision]
    models = [fitted[k][1]['model'] / fitted[k][1]['baseline'] for k in selected]
    B = float(np.mean([fitted[k][1]['baseline'] for k in selected]))
    norm = np.mean(np.stack(models), axis=0)
    final = norm * B
    valid = np.logical_or.reduce([fitted[k][0]['valid'] for k in selected])
    if decision == 'both':
        valid &= ~discrepancy
    measured_proxy = np.full(len(t), np.nan)
    values = []
    for k in selected:
        obs = fitted[k][0]
        scaled = obs['anchors'] * (B / fitted[k][1]['baseline'])
        z = np.full(len(t), np.nan)
        db = float(np.median(np.diff(obs['bin_time'])))
        bid = np.clip(np.floor((t - t[0]) / db).astype(int), 0, len(scaled) - 1)
        z[obs['valid']] = scaled[bid[obs['valid']]]
        values.append(z)
    tmp = np.stack(values)
    ct = np.isfinite(tmp).sum(axis=0)
    np.divide(np.nansum(tmp, axis=0), ct, out=measured_proxy, where=ct > 0)
    return dict(
        final=final,
        normalized=norm,
        accepted={k: k in selected for k in ['Left', 'Right']},
        decision=decision,
        valid=valid,
        interpolated=~valid,
        disagreement=discrepancy,
        manual_review_disagreement=bool(discrepancy.mean() > 0.04),
        combined=measured_proxy,
        baseline_scale_factors={k: B / fitted[k][1]['baseline'] for k in selected},
        eye_aligned={k: fitted[k][1]['model'] * B / fitted[k][1]['baseline'] for k in selected},
        model_fit_uncertain=any((fitted[k][1]['fit_uncertain'] for k in selected)),
        model_qualities={k: quality[k] for k in quality},
    )

def fit_light_phase_from_measurements(left: dict, right: dict, leftfit: dict, rightfit: dict, combined: dict, t: np.ndarray, ph: Phases) -> dict:
    """Estimate the light phase from accepted short-window raw measurements."""
    original = np.asarray(combined['final'], dtype=float)
    light = (t >= ph.light_start) & (t < ph.light_end)
    grid = t[light] - ph.light_start
    empty_light = original.copy()
    empty_light[light] = np.nan
    output = dict(
        signal=empty_light,
        estimated=False,
        valid=False,
        reason='no_measurement_supported_light_fit',
        warnings=[],
        selected_eye=combined['decision'],
        light_anchor_fraction=0.0,
        longest_unobserved_s=np.nan,
        median_relative_fit_error=np.nan,
        outlier_fraction=np.nan,
        peak_velocity_time_s=np.nan,
        peak_velocity=np.nan,
        minimum_time_s=np.nan,
        minimum_area=np.nan,
        minimum_at_boundary=False,
        fitted_midpoint_s=np.nan,
        fitted_width_s=np.nan,
        light_model='none',
    )
    if len(grid) < 10:
        output['reason'] = 'light_recording_too_short'
        return output
    selected = [k for k in ('Left', 'Right') if combined['accepted'][k]]
    tracks = {'Left': left, 'Right': right}
    fits = {'Left': leftfit, 'Right': rightfit}
    selected = [s for s in selected if fits[s]['usable'] and np.isfinite(fits[s]['baseline']) and (fits[s]['baseline'] > 0)]
    if not selected:
        output['reason'] = 'no_usable_eye'
        return output
    pre = np.flatnonzero(t < ph.light_start)
    post = np.flatnonzero(t >= ph.light_end)
    if not len(pre) or not len(post):
        output['reason'] = 'phase_boundaries_missing'
        return output
    start = float(original[pre[-1]])
    end = float(original[post[0]])
    if not np.isfinite(start) or start <= 0 or (not np.isfinite(end)) or (end <= 0):
        output['reason'] = 'invalid_baseline_or_recovery'
        return output
    duration = float(ph.light_duration_s)
    rel = []
    observed = []
    weights = []
    ref = tracks[selected[0]]
    ref_times = ref['bin_time']
    candidates_bin = (ref_times >= ph.light_start + 0.02) & (ref_times < ph.light_end - 0.02)
    bin_rels = ref_times[candidates_bin] - ph.light_start
    bin_seen = np.zeros(len(bin_rels), dtype=bool)
    for i, bt in enumerate(ref_times[candidates_bin]):
        choices = []
        for side in selected:
            tr = tracks[side]
            jj = int(np.argmin(np.abs(tr['bin_time'] - bt)))
            if abs(tr['bin_time'][jj] - bt) > 0.04 or not tr['bin_quality'][jj] or tr['bin_occupancy'][jj] < 0.65 or (not np.isfinite(tr['anchors'][jj])):
                continue
            v = float(tr['anchors'][jj]) * start / float(fits[side]['baseline'])
            choices.append((v, float(tr['bin_occupancy'][jj])))
        if len(choices) == 2 and abs(choices[0][0] - choices[1][0]) > 0.16 * start:
            continue
        if not choices:
            continue
        bin_seen[i] = True
        v = float(np.average([p[0] for p in choices], weights=[p[1] for p in choices]))
        if bt - ph.light_start >= 1.75:
            rel.append(float(bt - ph.light_start))
            observed.append(v / start)
            weights.append(float(np.mean([p[1] for p in choices])))
    early_end = min(1.8, 0.4 * duration)
    edges = np.arange(0.0, early_end + 1e-08, 0.08)
    early_seen = []
    for aa, bb in zip(edges[:-1], edges[1:]):
        choices = []
        window = (t >= ph.light_start + aa) & (t < ph.light_start + bb)
        for side in selected:
            tr = tracks[side]
            vv = np.asarray(tr['raw'][window], float)
            vv = vv[np.isfinite(vv) & (vv > 0)]
            if len(vv) < 8:
                continue
            order = np.sort(vv)
            counts = np.searchsorted(order, order + 0.075 * fits[side]['baseline'], side='right') - np.arange(len(order))
            k = int(np.argmax(counts))
            last = k + int(counts[k])
            occ = float(counts[k] / len(order))
            if occ < 0.7:
                continue
            med = float(np.median(order[k:last])) * start / float(fits[side]['baseline'])
            choices.append((med, occ))
        if len(choices) == 2 and abs(choices[0][0] - choices[1][0]) > 0.16 * start:
            continue
        if not choices:
            continue
        v = float(np.average([v for v, _ in choices], weights=[q for _, q in choices]))
        rel.append(float((aa + bb) / 2))
        observed.append(v / start)
        weights.append(0.5 * float(np.mean([q for _, q in choices])))
        early_seen.append(float((aa + bb) / 2))
    x = np.asarray(rel, float)
    y = np.asarray(observed, float)
    w = np.asarray(weights, float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]
    if len(x) < 5:
        output['reason'] = 'fewer_than_five_measured_anchors'
        return output
    coverage = float(bin_seen.mean()) if len(bin_seen) else 0.0
    max_gap = float(max_run_seconds(~bin_seen, 5.0)) if len(bin_seen) else duration
    if len(edges) > 1:
        centers = (edges[:-1] + edges[1:]) / 2
        shortseen = np.array([np.any(np.abs(np.asarray(early_seen) - c) < 0.001) for c in centers])
        max_gap = max(max_gap, float(max_run_seconds(~shortseen, 12.5)))
    warnings = []
    if any((fits[s].get('baseline_from_earlier_window', False) for s in selected)):
        warnings.append('baseline_2s_not_measured')
    if coverage < 0.7 or max_gap > 0.65 or len(x) < 18:
        warnings.append('low_light_anchor_coverage')
    if np.count_nonzero(x < 1.3) < 7 or np.min(x) > 0.16:
        warnings.append('early_kinetics_not_observed')
    if np.count_nonzero(x > duration - 1.2) < 3:
        warnings.append('late_light_not_observed')
    output.update(light_anchor_fraction=round(coverage, 4), longest_unobserved_s=round(max_gap, 3))
    escape_start = min(2.3, 0.4 * duration)

    def model(tt, p):
        depth, lag, tau, shape, escape = p
        zz = np.maximum((tt - lag) / tau, 0.0)
        cumulative = -np.expm1(-np.minimum(zz ** shape, 700.0))
        frac = np.clip((tt - escape_start) / max(duration - escape_start, 0.2), 0.0, 1.0)
        return 1.0 - depth * cumulative + escape * (frac * frac * (3 - 2 * frac))

    def residual(p):
        return np.r_[np.sqrt(w) * (model(x, p) - y), 0.45 * (model(np.array([duration]), p)[0] - end / start), 0.1 * p[4]]
    depth_est = float(np.clip(1 - np.nanquantile(y, 0.2), 0.09, 0.92))
    bounds = ([0.045, 0.0, 0.095, 1.35, -0.05], [0.96, min(0.8, duration * 0.35), 2.2, 3.3, 0.35])
    sols = []
    for lag in (0.005, 0.1, 0.22, 0.38, 0.65):
        for tau in (0.18, 0.4, 0.85):
            for shape in (1.65, 2.5):
                seed = np.array([depth_est, lag, tau, shape, 0.03])
                seed = np.clip(seed, np.array(bounds[0]) + 0.0001, np.array(bounds[1]) - 0.0001)
                try:
                    z = least_squares(residual, seed, bounds=bounds, loss='soft_l1', f_scale=0.045, max_nfev=145)
                    if np.isfinite(z.x).all():
                        err = np.abs(model(x, z.x) - y)
                        trim = int(max(4, np.ceil(0.85 * len(err))))
                        score = float(np.mean(np.sort(err)[:trim]))
                        sols.append((score, z.x))
                except (ValueError, FloatingPointError):
                    continue
    if not sols:
        output['reason'] = 'nonlinear_fit_failed'
        return output
    score, pars = min(sols, key=lambda zz: zz[0])
    r = np.abs(model(x, pars) - y)
    med = float(np.median(r))
    outfrac = float(np.mean(r > 0.12))
    yhat = model(grid, pars) * start
    fitted_end = float(model(np.array([duration]), pars)[0] * start)
    delta = end - fitted_end
    tailw = min(1.0, 0.18 * duration)
    blend = np.clip((grid - (duration - tailw)) / tailw, 0, 1)
    yhat += delta * blend * blend * (3 - 2 * blend)
    if med > 0.075 or outfrac > 0.2:
        warnings.append('incoherent_or_outlying_anchors')
    if abs(delta) > 0.07 * start:
        warnings.append('light_recovery_boundary_discrepancy')
    if pars[1] > bounds[1][1] - 0.04:
        warnings.append('estimated_reaction_lag_near_upper_bound')
    if pars[2] < bounds[0][2] + 0.015 or pars[2] > bounds[1][2] - 0.07:
        warnings.append('constriction_time_scale_near_bound')
    if pars[3] < bounds[0][3] + 0.04 or pars[3] > bounds[1][3] - 0.04:
        warnings.append('constriction_shape_near_bound')
    d = np.gradient(yhat, grid)
    search = (grid >= 0.04) & (grid < min(3.8, duration - 0.2))
    if not np.any(search):
        output['reason'] = 'derivative_window_unavailable'
        return output
    inds = np.flatnonzero(search)
    j = int(inds[np.argmin(d[search])])
    mi = int(np.argmin(yhat))
    if not np.isfinite(d[j]) or d[j] >= 0:
        warnings.append('no_supported_constriction')
    if min(abs(x - grid[j])) > 0.17:
        warnings.append('mcv_not_directly_observed')
    if grid[mi] > duration - 0.3:
        warnings.append('minimum_at_light_end_boundary')
    signal = original.copy()
    signal[light] = yhat
    assert np.array_equal(signal[~light], original[~light], equal_nan=True)
    output.update(
        signal=signal,
        estimated=True,
        valid=not warnings,
        warnings=warnings,
        reason='high_confidence_robust_light_fit' if not warnings else 'model_estimate_with_qc_warnings',
        median_relative_fit_error=round(med, 5),
        outlier_fraction=round(outfrac, 4),
        fitted_midpoint_s=float(pars[1]),
        fitted_width_s=float(pars[2]),
        peak_velocity_time_s=float(grid[j]),
        peak_velocity=float(d[j]),
        minimum_area=float(yhat[mi]),
        minimum_time_s=float(grid[mi]),
        minimum_at_boundary=bool(grid[mi] > duration - 0.3),
        light_model='variable_lag_shape_rise_no_fixed_latency',
    )
    return output

def analyze(path: Path, output_root: Path, output_label: str | None=None) -> dict:
    """Process one EyeSeeCam recording and write model and QC outputs."""
    path = Path(path)
    hdr = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(INPUT_COLUMNS) - set(hdr))
    if missing:
        raise ValueError('Fehlende CSV-Spalten: ' + ', '.join(missing))
    frame = pd.read_csv(path, usecols=list(INPUT_COLUMNS), dtype={'Time': str})
    raw_time_strings = frame['Time'].to_numpy(copy=True)
    for col in INPUT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    ph = determine_phases(frame)
    t = frame['Time'].to_numpy(float)
    label = output_label or subject_label(path)
    target = output_root / label
    target.mkdir(parents=True, exist_ok=True)
    lraw = get_area_proxy(frame, 'Left')
    rraw = get_area_proxy(frame, 'Right')
    left = reconstruct_eye(lraw, t, ph)
    right = reconstruct_eye(rraw, t, ph)
    leftfit = fit_eye_physiology(left, t, ph)
    rightfit = fit_eye_physiology(right, t, ph)
    combined = combine_physiology_models(left, right, leftfit, rightfit, t, ph)
    reference_baseline_recovery = combined['final'].copy()
    light_diagnostic = fit_light_phase_from_measurements(left, right, leftfit, rightfit, combined, t, ph)
    final = light_diagnostic['signal']
    interp = combined['interpolated']
    merged_raw = combined['combined']
    events = _potential_tracking_events(final, t, ph, float(np.nanmean([z['baseline'] for z in (left, right) if z['usable']])))
    reconstructed_global_pct, qc_overall = reconstruction_qc(interp)
    qc = {}
    for name, (start, end) in phase_windows(ph).items():
        q = window_qc(t, merged_raw, interp, final, start, end, ph.fs)
        if q['status'] == 'NOT_RECORDED':
            qc[name] = q
            continue
        w = window_mask(t, start, end)
        events_f = float(events[w].mean())
        warnings = []
        if q['original_fraction'] is not None and q['original_fraction'] < 0.25:
            warnings.append('low_original_support')
        if events_f > 0.06:
            warnings.append('abrupt_tracking_events')
        if combined['model_fit_uncertain']:
            warnings.append('model_fit_uncertain')
        if q['remaining_nan_s'] and q['remaining_nan_s'] > 0.08:
            warnings.append('unfilled_signal_gap')
        if combined['manual_review_disagreement']:
            warnings.append('interocular_disagreement')
        if q['interpolated_fraction'] is not None and q['interpolated_fraction'] > 0.25:
            warnings.append('window_reconstruction_above_25pct_INFORMATION_ONLY')
        q['warning_flags'] = ';'.join(warnings)
        q['tracking_jump_fraction'] = round(events_f, 4)
        qc[name] = q
    reference_features = features(t, reference_baseline_recovery, ph, qc)
    feat = features(t, final, ph, qc)
    if not light_diagnostic['estimated']:
        for key in ['Constriction_minimum_area_proxy', 'Constriction_minimum_normalized', 'Time_to_minimum_s', 'Maximum_Constriction_Velocity_proxy_per_s', 'Maximum_Constriction_Velocity_time_s']:
            feat[key] = float('nan')
        for key in list(feat):
            if key.startswith('Redilation_AUC_from_minimum_') or key.startswith('Redilation_AUC_fractional_recovery_'):
                feat[key] = float('nan')
    for key in feat:
        if key.startswith('Redilation_AUC_') and 'minimum' not in key and ('fractional' not in key):
            a, b = (feat[key], reference_features[key])
            if np.isfinite(a) and np.isfinite(b) and (not np.isclose(a, b, rtol=0, atol=1e-10)):
                raise AssertionError(f'Classical AUC changed after fitting the light phase: {key}: {a} / {b}')
    result = {
        'Subject': label,
        'Source_file': path.name,
        'Sampling_rate_Hz': round(ph.fs, 3),
        'Light_duration_s': round(ph.light_duration_s, 3),
        'Recovery_recorded_s': round(ph.recovery_duration_s, 3),
        'QC_Overall': qc_overall,
        'QC_Rule': 'global_reconstructed_percent_gte_25_FAIL',
        'QC_Threshold_reconstructed_percent': GLOBAL_RECONSTRUCTION_EXCLUSION_PERCENT,
        'QC_Left_used': combined['accepted']['Left'],
        'QC_Right_used': combined['accepted']['Right'],
        'QC_Selected_eye': combined['decision'],
        'QC_Physiology_model': 'baseline_recovery_plus_robust_light',
        'QC_Light_fit_valid': light_diagnostic['valid'],
        'QC_Light_estimated_latency_s': light_diagnostic['fitted_midpoint_s'],
        'QC_Light_estimated_time_scale_s': light_diagnostic['fitted_width_s'],
        'QC_Light_fit_estimated': light_diagnostic['estimated'],
        'QC_Light_fit_warning_flags': ';'.join(light_diagnostic['warnings']),
        'QC_Light_fit_reason': light_diagnostic['reason'],
        'QC_Light_anchor_fraction': light_diagnostic['light_anchor_fraction'],
        'QC_Light_max_unsupported_s': light_diagnostic['longest_unobserved_s'],
        'QC_Light_median_relative_fit_error': light_diagnostic['median_relative_fit_error'],
        'QC_Light_outlier_fraction': light_diagnostic['outlier_fraction'],
        'QC_Light_minimum_at_boundary': light_diagnostic['minimum_at_boundary'],
        'Area_unit': 'EyeSeeCam_covariance_area_proxy_not_mm2',
        'Feature_source': 'PLR_Zeitreihe.csv / Analysis_signal_smoothed',
        'QC_Model_fit_uncertain': combined['model_fit_uncertain'],
        'QC_Left_median_relative_fit_error': leftfit['median_relative_residual'],
        'QC_Left_baseline_suspect': leftfit.get('baseline_suspect', True),
        'QC_Right_median_relative_fit_error': rightfit['median_relative_residual'],
        'QC_Right_baseline_suspect': rightfit.get('baseline_suspect', True),
        'QC_Left_anchor_fraction': round(left['anchor_bin_fraction'], 3),
        'QC_Right_anchor_fraction': round(right['anchor_bin_fraction'], 3),
        'QC_Left_longest_unsupported_s': round(left['longest_gap_s'], 3),
        'QC_Right_longest_unsupported_s': round(right['longest_gap_s'], 3),
        'QC_Interocular_disagreement_percent': round(100 * combined['disagreement'].mean(), 2),
        'QC_Interpolated_percent': round(reconstructed_global_pct, 4),
        'QC_Longest_interpolation_s': round(max_run_seconds(interp, ph.fs), 3),
        'QC_Remaining_unfilled_gap_s': round(max_run_seconds(~np.isfinite(final), ph.fs), 3),
        **{f'QC_{key}': value['status'] for key, value in qc.items()},
        **feat,
    }
    usedfits = [leftfit if combined['accepted']['Left'] else None, rightfit if combined['accepted']['Right'] else None]
    baseline_fallback = any((v and v.get('baseline_from_earlier_window', False) for v in usedfits))
    result['QC_Baseline_fallback_from_earlier_window'] = bool(baseline_fallback)
    result['MODEL_Baseline_reference_area_proxy'] = feat['Baseline_mean_area_proxy_2s']
    if baseline_fallback:
        result['Baseline_mean_area_proxy_2s'] = np.nan
    for name, windows in {'AUC_0_10': ('baseline_pre_2s', 'recovery_0_10s'), 'AUC_10_20': ('baseline_pre_2s', 'recovery_10_20s'), 'AUC_20_30': ('baseline_pre_2s', 'recovery_20_30s'), 'AUC_0_20': ('baseline_pre_2s', 'recovery_0_10s', 'recovery_10_20s'), 'AUC_0_30': ('baseline_pre_2s', 'recovery_0_10s', 'recovery_10_20s', 'recovery_20_30s')}.items():
        result['QC_Eligible_' + name] = bool(qc_overall == 'PASS' and (not baseline_fallback) and all((qc[k]['status'] != 'NOT_RECORDED' for k in windows)))
    result['QC_Eligible_MCV'] = bool(qc_overall == 'PASS' and light_diagnostic['estimated'] and np.isfinite(feat.get('Maximum_Constriction_Velocity_proxy_per_s', np.nan)))
    result['QC_Eligible_minimum'] = bool(qc_overall == 'PASS' and light_diagnostic['estimated'] and np.isfinite(feat.get('Constriction_minimum_area_proxy', np.nan)))
    result_safe = result.copy()
    for a, b in [('0_10', 'AUC_0_10'), ('10_20', 'AUC_10_20'), ('20_30', 'AUC_20_30'), ('0_20', 'AUC_0_20'), ('0_30', 'AUC_0_30')]:
        if not result['QC_Eligible_' + b]:
            for prefix in ['Redilation_AUC_', 'Redilation_AUC_from_minimum_', 'Redilation_AUC_fractional_recovery_']:
                result_safe[prefix + a + '_s'] = np.nan
    if not result['QC_Eligible_minimum']:
        for name in ('0_10', '10_20', '20_30', '0_20', '0_30'):
            for pref in ('Redilation_AUC_from_minimum_', 'Redilation_AUC_fractional_recovery_'):
                result_safe[pref + name + '_s'] = np.nan
    if baseline_fallback:
        result_safe['Baseline_mean_area_proxy_2s'] = np.nan
    if not result['QC_Eligible_MCV']:
        for v in ['Constriction_minimum_area_proxy', 'Constriction_minimum_normalized', 'Time_to_minimum_s', 'Maximum_Constriction_Velocity_proxy_per_s', 'Maximum_Constriction_Velocity_time_s']:
            result_safe[v] = np.nan
    ts = pd.DataFrame({
        'Time_s': raw_time_strings,
        'Time_relative_light_on_s': t - ph.light_start,
        'Time_relative_light_off_s': t - ph.light_end,
        'Phase_code': frame['visual_pos3'].to_numpy(),
        'Left_area_proxy_raw': lraw,
        'Right_area_proxy_raw': rraw,
        'Left_reconstructed': left['trace'],
        'Right_reconstructed': right['trace'],
        'Left_accepted_anchor': left['valid'].astype(int),
        'Right_accepted_anchor': right['valid'].astype(int),
        'Interocular_disagreement': combined['disagreement'].astype(int),
        'Combined_before_interpolation': merged_raw,
        'Combined_long_gap_interpolated': interp.astype(int),
        'Analysis_signal_smoothed': final,
        'Light_phase_robust_fit_used': ((t >= ph.light_start) & (t < ph.light_end) & light_diagnostic['estimated']).astype(int),
        'Analysis_signal_normalized': final / feat['Baseline_mean_area_proxy_2s'] if np.isfinite(feat['Baseline_mean_area_proxy_2s']) and feat['Baseline_mean_area_proxy_2s'] > 0 else np.full(len(final), np.nan),
        'Flag_abrupt_poststimulus_change': events.astype(int),
        })
    ts.to_csv(target / 'PLR_Zeitreihe.csv', index=False, float_format='%.17g')
    pd.DataFrame([result]).to_csv(target / 'PLR_Kennwerte_MODELL.csv', index=False)
    pd.DataFrame([result_safe]).to_csv(target / 'PLR_Kennwerte_QC.csv', index=False)
    lightfig, ax = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
    wlight = (t >= ph.light_start - 0.7) & (t <= ph.light_end + 0.8)
    for side, tr, fit, c in [('Left', left, leftfit, '#a68a39'), ('Right', right, rightfit, '#549a9a')]:
        accepted = tr['bin_quality'] & (tr['bin_time'] >= ph.light_start) & (tr['bin_time'] < ph.light_end)
        if np.any(accepted):
            ax.scatter(tr['bin_time'][accepted] - ph.light_start, tr['anchors'][accepted], s=16, alpha=0.5, c=c, label=side + ' accepted 200-ms bins')
    ax.plot(t[wlight] - ph.light_start, final[wlight], lw=2.2, color='#205d92', label='Robustly fitted light-phase curve')
    if light_diagnostic['estimated']:
        ax.axvline(light_diagnostic['peak_velocity_time_s'], color='#b64646', lw=0.9, ls='--', label='MCV time (individually fitted)')
        ax.scatter([light_diagnostic['minimum_time_s']], [light_diagnostic['minimum_area']], s=55, marker='x', color='#6b37a7', label='Light minimum')
    ax.axvline(0, color='#ad801e', ls=':', lw=1)
    ax.axvline(ph.light_duration_s, color='#ad801e', ls=':', lw=1)
    ax.set_xlabel('Zeit seit Lichtbeginn (s)')
    ax.set_ylabel('Pupillenflaechen-Proxy')
    ax.set_title(label + ' | Lichtphase: ' + light_diagnostic['reason'])
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.14)
    lightfig.savefig(target / 'PLR_Lichtphase_Qualitaetscheck.png', dpi=155)
    plt.close(lightfig)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    xx = t - ph.light_start
    for ax, (name, data) in zip(axes[:2], [('Links', left), ('Rechts', right)]):
        ax.plot(xx, data['raw'], lw=0.3, alpha=0.22, color='#818991', label='Roh')
        fitted = leftfit if name == 'Links' else rightfit
        if fitted['usable']:
            without_light = fitted['model'].copy()
            without_light[(t >= ph.light_start) & (t < ph.light_end)] = np.nan
            ax.plot(xx, without_light, lw=2.0, color='#086b91', label='Baseline / recovery model')
        ax.scatter(data['bin_time'][data['bin_quality']] - ph.light_start, data['anchors'][data['bin_quality']], s=4, alpha=0.5, label='Akzeptierte Anker')
        ax.set_ylabel(name + ' Fläche')
        ax.legend(fontsize=8, loc='upper right')
    axes[2].plot(xx, final, lw=1.6, color='#27749a', label='Final: ' + combined['decision'])
    axes[2].fill_between(xx, final, where=interp, color='tab:orange', alpha=0.26, label='Rekonstruiert (keine Messung)')
    axes[2].set_ylabel('Finaler Proxy')
    axes[2].set_xlabel('Sekunden relativ zu Lichtbeginn')
    axes[2].legend(fontsize=8, loc='upper right')
    for ax in axes:
        ax.axvspan(0, ph.light_duration_s, color='#f1bd67', alpha=0.22)
        ax.grid(alpha=0.15)
    fig.suptitle(f'{label} | PhysioFit | Gesamtstatus {qc_overall}', fontsize=13)
    fig.savefig(target / 'PLR_QC_Vergleich.png', dpi=145)
    plt.close(fig)
    write_final_analysis_figure(target / 'PLR_Finalkurve.png', t, ph, final, feat, qc, label + (' | MODELLSCHAETZUNG, QC pruefen' if qc_overall != 'PASS' else ' | MODELLFIT'))
    qc_report = {
        'Algorithm_version': '5.8',
        'Phases': asdict(ph),
        'Selected_eye': combined['decision'],
        'QC_Overall': qc_overall,
        'QC_global_reconstructed_percent': round(reconstructed_global_pct, 4),
        'QC_rule': 'FAIL when >=25% of the full recording lacks accepted measurement anchors; other warnings are informational',
        'Physiology_fit_left': {k: v for k, v in leftfit.items() if k not in ('model', 'bin_residual')},
        'Physiology_fit_right': {k: v for k, v in rightfit.items() if k not in ('model', 'bin_residual')},
        'Window_QC': qc,
        'Light_phase_fit': {k: v for k, v in light_diagnostic.items() if k != 'signal'},
        'Measured_proxy': 'sqrt(CovCol*CovRow - CovColRow^2); uncalibrated area proxy',
        'Method': 'Baseline and recovery model retained; light-phase fit uses accepted 80-ms and 200-ms raw-data anchors. All features derive from Analysis_signal_smoothed.',
        'Limitation': 'Global QC is solely a >=25% reconstruction threshold, not a complete quality validation. MCV is a model-derived estimate.',
    }
    (target / 'PLR_QC_Protokoll.json').write_text(
        json.dumps(qc_report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return result_safe

def batch(files: list[Path], out: Path, verbose=True) -> tuple[pd.DataFrame, list[str]]:
    """Run the same analysis for each input file and aggregate the outputs."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    used_names = {}
    for i, file in enumerate(files, 1):
        base = subject_label(file)
        used_names[base] = used_names.get(base, 0) + 1
        label = base if used_names[base] == 1 else f'{base}__file{used_names[base]}'
        try:
            result = analyze(file, out, label)
            rows.append(result)
            if verbose:
                print(f"[{i}/{len(files)}] OK {label}: {result['QC_Overall']}", flush=True)
        except Exception as e:
            errors.append(f'{file}: {type(e).__name__}: {e}')
            if verbose:
                print(f'[{i}/{len(files)}] ERROR {file.name}: {e}', flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / 'PLR_Alle_Probanden_QC.csv', index=False)
    model_rows = []
    for safe_row in rows:
        subject = str(safe_row['Subject'])
        model_file = out / subject / 'PLR_Kennwerte_MODELL.csv'
        if model_file.is_file():
            model_df = pd.read_csv(model_file)
            if len(model_df) == 1:
                model_rows.append(model_df)
    models = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    models.to_csv(out / 'PLR_Alle_Probanden_MODELL.csv', index=False)
    (out / 'PLR_Fehlerprotokoll.txt').write_text('\n'.join(errors) + '\n' if errors else '', encoding='utf8')
    (out / 'METHODEN_QC.txt').write_text(
        """EyeSeeCam PLR PhysioFit | Algorithm 5.8
Cohort exclusion: reconstruction fraction >=25% of the entire recording.
Reconstruction = samples without accepted 200-ms measurement-bin support,
including interocular disagreement where both eyes were selected.
Phase RECORDED/NOT_RECORDED statuses and fit warnings are not exclusions.
Parameters without sufficient observed support remain missing (NaN).
PLR_Kennwerte_MODELL.csv: model estimates regardless of global QC.
PLR_Kennwerte_QC.csv: values masked by global QC and availability.
AUC intervals are seconds after light offset; area is a covariance proxy.
Signal models and automatic QC thresholds have not been externally validated.
""",
        encoding='utf-8',
    )
    return (tab, errors)

def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    root = tk.Tk()
    root.title('EyeSeeCam PLR PhysioFit')
    root.geometry('900x660')
    container = ttk.Frame(root, padding=14)
    container.pack(fill='both', expand=True)
    ttk.Label(container, text='EyeSeeCam PLR PhysioFit – Analyse', font=('Arial', 15, 'bold')).pack(anchor='w')
    ttk.Label(container, text='Ausschluss nur bei mindestens 25 % rekonstruierten Zeitpunkten in der gesamten Aufnahme.', wraplength=800).pack(anchor='w', pady=(4, 10))
    filebox = tk.Listbox(container, height=13, selectmode='extended')
    filebox.pack(fill='both', expand=True)
    files = []

    def add_files():
        paths = filedialog.askopenfilenames(filetypes=[('CSV', '*.csv')])
        for name in paths:
            p = Path(name)
            if p not in files:
                files.append(p)
                filebox.insert('end', p.name)

    def clear_files():
        files.clear()
        filebox.delete(0, 'end')
    toolbar = ttk.Frame(container)
    toolbar.pack(fill='x', pady=5)
    ttk.Button(toolbar, text='CSV-Dateien hinzufuegen', command=add_files).pack(side='left')
    ttk.Button(toolbar, text='Liste leeren', command=clear_files).pack(side='left', padx=7)
    folder = tk.StringVar(value=str(Path.home() / 'EyeSeeCam_PLR_QC'))
    ttk.Entry(container, textvariable=folder).pack(fill='x', pady=4)

    def choose_dir():
        d = filedialog.askdirectory()
        if d:
            folder.set(d)
    ttk.Button(container, text='Ausgabeordner waehlen', command=choose_dir).pack(anchor='w')
    status = tk.StringVar(value='Noch keine Analyse gestartet')
    ttk.Label(container, textvariable=status, wraplength=800).pack(anchor='w', pady=6)

    def start():
        if not files:
            messagebox.showwarning('Fehler', 'Bitte CSV-Dateien auswaehlen')
            return
        import threading
        btn.configure(state='disabled')
        selected = list(files)
        dest = Path(folder.get()).expanduser()
        status.set(f'Analysiere {len(selected)} Dateien ...')

        def work():
            try:
                tab, errors = batch(selected, dest, verbose=False)
                info = f'Fertig: {len(tab)} erfolgreich, {len(errors)} Fehler. Ordner: {dest}'
                root.after(0, lambda: (status.set(info), btn.configure(state='normal'), messagebox.showinfo('Fertig', info)))
            except Exception as exc:
                msg = str(exc)
                root.after(0, lambda: (status.set('Fehler: ' + msg), btn.configure(state='normal'), messagebox.showerror('Fehler', msg)))
        threading.Thread(target=work, daemon=True).start()
    btn = ttk.Button(container, text='Analyse starten', command=start)
    btn.pack(anchor='w', pady=10)
    root.mainloop()

def main() -> int:
    parser = argparse.ArgumentParser(description='EyeSeeCam PLR analysis; only cohort exclusion: >=25% reconstructed samples')
    parser.add_argument('--batch', nargs='+', type=Path, help='EyeSeeCam CSV Dateien')
    parser.add_argument('--output', type=Path, default=Path.cwd() / 'PLR_QC_Ergebnisse')
    args = parser.parse_args()
    if not args.batch:
        launch_gui()
        return 0
    tab, errors = batch(args.batch, args.output)
    print(f'Erfolgreich: {len(tab)} | Fehler: {len(errors)} | Ausgabe: {args.output}')
    return 1 if errors else 0
if __name__ == '__main__':
    sys.exit(main())
