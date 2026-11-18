# Task 2 plan: RUL regression, following standard practice

This plan follows how RUL regression is usually done in the research literature. Every step says:
- **Common practice:** what researchers usually do.
- **Why:** the reason, so you can explain it in the oral exam.
- **Our version:** what we do on our data, and anything we change from the usual recipe.
- **Check:** how we know the step worked, and what goes in the report.

One idea at a time, in order. No step uses something that hasn't been explained in an earlier step.

---

## 0. Where this task comes from (read this first)

Our dataset is a copy of a well-known benchmark: **NASA C-MAPSS** (simulated turbofan engines; Saxena et al., 2008). The match is exact:

| C-MAPSS | Our data |
|---|---|
| 3 operational settings, 21 sensors per cycle | `setel_1..3`, `s01..s21` |
| Training engines run until failure | `rul_train`: machines run until failure |
| Test engines stop at a random point; predict RUL at the last cycle | `rul_test`: same |
| Scoring function: exp(−d/13)−1 if early, exp(d/10)−1 if late | The exam's asymmetric score, same formula |
| Subsets FD002/FD004: 6 operating conditions; FD003/FD004: 2 failure modes | We found 6 regimes and 2 failure modes, so we're FD004-like |

**Why this matters:** there are hundreds of papers on C-MAPSS, so there is a well-established standard pipeline. We follow it instead of inventing our own. When the lecturer asks "why did you do X?", the answer is usually "this is the standard approach, and here is the reason behind it".

The standard pipeline has the same six parts as the exam:

```
raw logs → (a) clean + normalize → (b) make RUL labels → (c) cut into windows
        → (d) train models → (e) validate per machine → (f) add uncertainty → predict test
```

---

## (a) Exploration and preprocessing: done

All five steps match the usual C-MAPSS practice:

| Our step | Common practice | Why |
|---|---|---|
| Cluster the 3 settings into 6 regimes, z-score each sensor *within its regime* | Standard for FD002/FD004: "operating-condition normalization" | The regime explains up to 99.9% of a sensor's variance. A global normalization would mostly encode the regime, not wear. |
| Drop 7 sensors with no trend | Standard: most FD001 papers keep the same 14 informative sensors and drop the flat ones | Flat sensors are pure noise. Keeping them gives the model more ways to overfit. |
| Forward-fill missing readings | Not in original C-MAPSS (it has no gaps); forward-fill is the usual causal choice for time series | Interpolation would use the next reading, which is the future and doesn't exist at the last test cycle. |
| Remove spikes with a backward-looking median filter (Hampel filter) | Not in original C-MAPSS; Hampel is the standard robust outlier filter for signals | A spike is a sensor glitch, not wear. Backward-looking so it never uses the future. |
| Fit all statistics on training machines only | Universal ML practice | Otherwise information from validation or test machines leaks in. |

**Open item:** a gap at the very start of a history is filled with 0 (the regime average). A healthy machine usually sits slightly below or above 0, so the "healthy average" (the mean of training machines' first cycles) is a more sensible fill. This is a small fix; I'll do it at the start of (b).

Smoothing is common in papers (moving average or exponential smoothing). We don't smooth in preprocessing, because the deep models see a whole window and average out noise themselves. The non-deep baseline in (d) gets explicit window averages instead.

---

## (b) RUL labels

**Common practice: the piecewise-linear target** (Heimes, 2008; used by nearly every later paper).

For a training machine that fails at cycle `T`, the true remaining life at cycle `t` is `T − t`. Instead of using that directly, papers cap it:

```
RUL(t) = min(cap, T − t)
```

So the label is flat at `cap` early in life, then falls linearly to 0 at failure. Papers typically use a cap of 125–130; **the exam fixes it at 150**.

**Why cap it:**
- Early in life the machine is healthy and the sensors look the same whether it has 300 or 200 cycles left. Asking a model to tell those apart means asking it to predict something the data cannot show. It just learns noise.
- Fig. `t2_failure_modes` shows this directly: the curves are flat until roughly 100–150 cycles before failure.
- The cap also matches how the test is scored: any true RUL above 150 counts as 150.

**Our version:** exactly the formula above with cap 150, computed per machine from the training data.

**Check / report:**
- A plot of the label over one machine's life (flat, then linear).
- A histogram of all labels. Expect a large spike at 150, because many cycles are early-life. That spike matters for (c): we don't want the model to just learn "always say 150".
- One unit test: the label at each machine's last cycle is 0.

---

## (c) Sequences: sliding windows

**Common practice: fixed-length sliding windows** (Babu et al., 2016; Zheng et al., 2017; Li et al., 2018).

A model can't take a whole 300-cycle history in one go easily, and the test histories have different lengths. So we cut each history into overlapping windows of `W` consecutive cycles:

```
window ending at cycle t = readings from cycles (t−W+1 ... t)   →   label = RUL(t)
```

Slide `t` forward one cycle at a time to get one training sample per cycle. At test time we take the **single window ending at the last recorded cycle** of each machine.

**Why:**
- **No future leakage by construction:** a window ending at `t` only contains cycles up to `t`, and its label is the RUL at `t`. This is exactly the test situation: "here is the history so far, how long is left?"
- **Same input shape for every sample,** which is what CNNs and batched RNNs need.
- **Many training samples:** about 254k windows from 900 machines.

**Choosing W:** papers use 20–50 for C-MAPSS (often 30). It's a trade-off: a longer window shows more of the trend but leaves fewer usable cycles in short histories. Our shortest test history is **50 cycles**, so `W ≤ 50` means every test machine has a full window. We'll compare W = 30 and W = 50 in (d) instead of guessing.

**Histories shorter than W** (only the first few cycles of each training machine): the usual fix is to **pad on the left** by repeating the first reading. We can also simply start training windows at cycle `W`. We'll take the simpler option unless it hurts.

**Two extra inputs to decide on (common, but optional):**
1. **Cycle number (machine age):** it's known at test time, so it's legal, and some papers include it. It tells the model "this machine is already old".
2. **Regime one-hot:** after per-regime normalization the regime should carry no information. We'll check whether it still helps.

**Check / report:**
- Window counts.
- A diagram of one window and its label.
- A test that the last test window per machine ends exactly at its last recorded cycle.

---

## (d) Models: a non-deep baseline and at least 2 sequence architectures

The exam requires the main model to be deep learning, compared with a classical baseline. The usual order in papers is **simple → complex**, so every extra piece of complexity has to earn its place by beating the simpler model.

### d0. Non-deep baseline (the yardstick)
- **Common practice:** a regression model on hand-made window features, e.g. for each sensor: last value, window mean, and slope over the window.
- **Model:** Ridge regression (linear), plus one tree ensemble (random forest or gradient boosting).
- **Why:** if a deep model can't beat this, the deep model isn't learning anything useful. Trees also handle the two failure modes reasonably well (they can learn "if s13 drops OR s12 rises").

### d1. 1D-CNN (Babu et al., 2016; Li et al., 2018)
- **What it does:** small filters slide along the time axis of the window and detect local patterns (e.g. "this sensor is rising"). Stacking layers combines them; then a dense layer outputs one number, the RUL.
- **Why it suits RUL:** wear is a local, smooth trend. CNNs are fast and stable to train.

### d2. LSTM or GRU (Zheng et al., 2017)
- **What it does:** reads the window one cycle at a time, keeping a memory state. The final state goes to a dense layer that outputs RUL.
- **Why it suits RUL:** it's the classic sequence model. A GRU is a lighter LSTM with similar accuracy; we pick one and justify it.

### d3 (optional, only if time allows). TCN or a small Transformer
- A TCN is a CNN with dilated, causal convolutions (Bai et al., 2018); a Transformer uses self-attention (Vaswani et al., 2017).
- The exam only needs two architectures, so this is a bonus, not a requirement.

### Training setup (shared by all deep models, standard practice)
- **Loss:** mean squared error on the capped labels, since RMSE is the main metric.
- **Optimizer:** Adam with a learning-rate schedule and early stopping on the validation machines.
- **Sanity checks first:**
  - **Overfit one batch:** the loss should go to about 0. If it doesn't, there's a bug.
  - **Learning curves:** plot train vs validation loss over epochs to see under- or over-fitting.
- **Outputs** are clipped to [0, 150].
- **Seeds:** at least 3 per final model, reported as mean ± std.

### Ablations (small, cheap, good for the report)
- W = 30 vs 50
- With and without cycle number
- With and without dropped sensors (shows the selection in (a) was worth it)

**Check / report:** one table, model × (RMSE, score) on validation, mean ± std over seeds.

---

## (e) Validation: split by machine

**Common practice:**
- **Split by machine, never by row.** Rows from the same machine are almost identical neighbours. If machine 11's cycle 100 is in training and cycle 101 is in validation, the model just memorises machine 11 and the score looks far too good. This is the most common mistake in RUL papers.
- **Mimic the test:** papers evaluate on the *last cycle* of truncated histories, because that's what the test asks for. Evaluating on every window mixes in thousands of easy "RUL = 150" early-life windows and looks better than it is.

**Our version:**
1. Split the 900 training machines into 5 groups (GroupKFold). Each round trains on 4/5 and validates on the remaining 1/5.
   - Preprocessing and sensor selection are refitted on the training part every round, so validation machines stay unseen.
   - For speed, we may use one fixed 80/20 machine split to develop models, then 5-fold only for the final comparison.
2. For each validation machine, **cut its history at a random point**, the same way the test was made, and predict RUL at the cut.
   - We match the test's distribution: test histories have median 163 cycles vs 265 for full training lives.
   - We draw several cuts per machine for a stable estimate.
3. **Metrics:**
   - **RMSE.**
   - **Asymmetric score:** `d = predicted − true`. Late predictions (d > 0, the model says the machine has *more* life than it does) cost exp(d/10)−1. Early ones cost exp(−d/13)−1. Late is punished harder because a surprise failure is worse than servicing a bit too early.
   - The score is a *sum*, so it grows with the number of machines. We also report it per machine, so validation and test numbers are comparable.
4. **Per RUL range:** report both metrics for true RUL in [0–25], (25–50], (50–100], (100–150]. Errors near failure matter most for maintenance, and models are usually most accurate there.

**Check / report:**
- The metrics table.
- A plot of predicted vs true RUL on validation.
- An "error vs true RUL" plot.

---

## (f) Uncertainty and when to service

**Common practice:** two standard ways to give a deep model a prediction *interval* instead of a single number:
- **MC dropout** (Gal & Ghahramani, 2016): keep dropout switched on at prediction time and predict, say, 50 times. The spread of the 50 answers is the uncertainty. It's cheap: one trained model.
- **Deep ensemble** (Lakshminarayanan et al., 2017): train 5 copies with different seeds and use the spread of their predictions. It's usually better calibrated, and we already train several seeds for (d), so it costs almost nothing extra.

**Our version:**
1. Build 90% intervals with both methods on the validation machines.
2. **Check calibration (coverage):** what fraction of true RULs fall inside the 90% interval? If it's 90%, the intervals are honest. If it's 60%, they're too narrow. This is the step that makes uncertainty "honest", and it's often skipped in student work.
3. **When to service:**
   - A simple, explainable rule: service a machine when the **lower end** of its interval falls below the lead time needed to plan maintenance (e.g. 20 cycles). Using the lower bound is cautious, which matches the asymmetric score's dislike of late predictions.
   - Show on validation how many failures would be caught vs how much life is "wasted" by servicing early, for a few thresholds.
4. **Optional link to the score:** because late errors cost more, shifting predictions slightly down (e.g. predict the 40th percentile instead of the mean) can lower the asymmetric score. We test this on validation only, never on the test set.

**Check / report:**
- Coverage table.
- A plot of a few machines' predicted RUL over time with their intervals.
- The maintenance-rule trade-off.

---

## Final model and submission

- Pick the final model on **validation results only.** The test set is never used for any choice.
- Retrain on all 900 machines. If we use an ensemble, the 5 seed models are the ensemble.
- Predict at each test machine's last cycle, clip to [0, 150], and write `prediksi_rul.csv` with `id_unit,rul` (300 rows).
- `run_all.py --task 2` reproduces every number. Weights are saved in `kode/results/task2/` (small; RUL models are a few MB).

---

## Schedule (today 17 Nov, deadline 24 Nov 23:59)

| Date | Part | Stop for review? |
|---|---|---|
| 17 Nov | (b) labels + start-of-history fill fix | yes |
| 18 Nov | (c) windows + validation split from (e), built together because they share code | yes |
| 19 Nov | (d) baseline + first deep model + sanity checks | yes |
| 20 Nov | (d) second architecture + ablations | yes |
| 21 Nov | (e) full validation tables + (f) uncertainty | yes |
| 22 Nov | Final model, `prediksi_rul.csv`, report section | yes |
| 23 Nov | Buffer: merge with partner's Task 1, report PDF, repo + PRs | — |

**Also needed, outside Task 2:**
- **Task 3 (clustering)** has no owner yet and is worth 25 points.
- The **group GitHub repo** isn't set up.
- **LaTeX** (Overleaf or MiKTeX) isn't set up.

---

## References (check every entry with a citation checker before submitting)

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *Int. Conf. on Prognostics and Health Management (PHM)*.
- Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation. *Int. Conf. on Prognostics and Health Management (PHM)*.
- Ramasso, E., & Saxena, A. (2014). Performance benchmarking and analysis of prognostic methods for CMAPSS datasets. *Int. Journal of Prognostics and Health Management*.
- Babu, G. S., Zhao, P., & Li, X.-L. (2016). Deep convolutional neural network based regression approach for estimation of remaining useful life. *DASFAA*.
- Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). Long short-term memory network for remaining useful life estimation. *IEEE ICPHM*.
- Li, X., Ding, Q., & Sun, J.-Q. (2018). Remaining useful life estimation in prognostics using deep convolution neural networks. *Reliability Engineering & System Safety*, 172.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
