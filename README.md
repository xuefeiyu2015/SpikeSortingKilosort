# Spike Sorting Pipeline — Kilosort4 + Phy

This is the spike sorting pipeline for Neuropixel and Blackrock data in JLab.
The general procedural is as below:

```
run_sorting_pipeline.py             →           phy template-gui  →  run_exporting_pipeline.py
(main sorting procedural,needs a GPU)          (manuel curation)     (time alignment and remap)
```

## Start here

| If you are… | go to |
|---|---|
| a user with a session to sort, and you already know your sorting parameters | **[Step 1](#step-1--sort)**, then **[2](#step-2--curate-in-phy)** and **[3](#step-3--time-alignment-and-export)** |
| not sure about bad channels / whitening / drift, and want to see the recording first | **[Step 0](#step-0--optional-look-at-the-recording-first)** |
| installing this on a new computer | **[Step 4](#step-4--install-on-a-new-computer)** |
| missing a channel map (`probe_file`) | **[Step 5](#step-5--probe-files-channel-maps)** |
| staring at a `[--]` skipped or `[!!]` failed line | **[Step 6](#step-6--when-a-stage-skips-or-fails)** |

Three notebooks sit alongside the steps below, in `notebooks/`:

- **`run_pipeline_stepbystep.ipynb`** — if you want to understand what each step
  of the pipeline actually does, follow this one. Every cell calls a single verb,
  in the same order the two runner scripts call them, and shows what it returns.
  It is for reading and learning; for a real session use the scripts.
- **`setup_demo_data.ipynb`** — downloads the official Kilosort4 demo recording
  and its channel map, so you can run the whole pipeline start to finish without
  any rig data. Used in [Step 4f](#4f-prove-it-works).
- **`build_probe_config.ipynb`** — builds or inspects a channel map by hand.
  **Not needed in most cases**: a Neuropixels map comes from the run's own
  `.meta`, and a Utah array's comes from its `.cmp`. See
  [Step 5](#step-5--probe-files-channel-maps).

---

## Step 0 — optional: look at the recording first

**Skip this entirely if your sorting parameters are already settled.** It costs
nothing but your time, and it is the cheapest place to catch a dead bank of
channels or a drift assumption that does not hold.

```bash
conda activate kilosort4
python -m kilosort              # the Kilosort4 GUI
```

Load the binary and the probe file, and look at:

- **bad channels** — flat, saturated or noisy rows to list in `bad_channels`
- **the whitening / CAR effect** — the GUI shows raw, filtered and whitened
  traces side by side, which is the only honest way to decide `do_CAR`
- **drift** — whether the drift map justifies `nblocks: 1` (Neuropixels) or none
  at all (`nblocks: 0`, the Utah array — see the note in Step 1)

GUI guide: <https://kilosort.readthedocs.io/en/latest/gui_guide.html>

You *can* sort from the GUI too. Don't — the scripts below batch, log, and record
what they did in a session file you can read back six months later.

**Write what you learned into the session YAML.** These are the
keys it answers:

| key (under `kilosort:`) | what it decides |
|---|---|
| `bad_channels` | recording channel numbers to drop (rows of the binary, not array positions) |
| `do_CAR` | subtract the median across channels each batch (default true) |
| `highpass_cutoff` | Kilosort's own highpass, in Hz (default 300) |
| `nblocks` | drift correction: `0` off, `1` rigid |
| `artifact_threshold` | zero out batches above this amplitude |
| `tmin` / `tmax` | sort only this window of the binary, in seconds |(haven't implemented yet)


---
## Step 1 — preparation
Two cheaper checks in the same spirit, neither of which needs a GPU:

```bash
python tools/check_env.py --config configs/<session>.yaml   # env + inputs present
python run_sorting_pipeline.py --config configs/<session>.yaml --dry-run
```

`--dry-run` resolves the channel map, names the binary and prints the exact
settings Kilosort would get — without sorting. It is the way to check a session
file before booking rig time.

To watch the same thing happen one verb at a time, open
`notebooks/run_pipeline_stepbystep.ipynb`.



## Step 1 — sort

### 1a. Write the session config

I have listed 4 templates for the session cases, you can **COPY, PASTE, RENAME** according to your own sessions.

| template | Utah array | Neuropixels | aligns systems |
|---|---|---|---|
| `session_template_utah_probe.yaml` | spikes | spikes | yes |
| `session_template_1probe.yaml` | — (sync only) | spikes | yes |
| `session_template_2probes.yaml` | — (sync only) | spikes, **2 probes** | yes |
| `session_template_utah_only.yaml` | spikes | — (not recorded) | **no** |

for example, if I record only with 1 neuropixel, I will copy the `session_template_1probe.yaml` and rename it into `session_Porthos_1probe.yaml` . Then update the dir and file name to your folder that stored the .bin file from your recoridng. And the blackrock synchronize pulse dir and file names: *.ns5

Next, run: 

```bash
cp configs/session_template_1probe.yaml configs/session_Porthos_1probe.yaml
```

Here is one example for setting up the yaml file:

```yaml
blackrock_dir:   "Z:/Monkey Porthos/2026-08-13"     #dir pointing to the blackrock files, *ns6 or *ns5
neuropixels_dir: "Z:/Monkey Porthis/2026-08-13/Porthos_2026_08_13_g0/Porthos_2026_08_13_g0_imec0". #dir pointing to the neuropixel files
```

Three rules worth knowing before you edit:

- **The two systems must not share a directory.** Everything derived from a
  recording lands in `<system>_dir/kilosort4/`, so a shared folder means shared
  outputs. Loading raises if they collide. `neuropixels_dir` is the SpikeGLX
  *probe* folder (`…_g0_imec0`), not the session folder.
- **Whether a system recorded is derived from its block.** To say a system did
  not record, delete its block — do not set a flag. A missing share must stay
  "recorded, files missing", which is reported loudly, rather than "never
  recorded", which would skip in silence.
- **Above a `# ---------- defaults` fence is what you edit; below it is what you
  leave alone unless you have clear reason to change the default .** 

Two probes of one run are two `bin_files:` entries, with an optional `by_probe:`
block for per-probe `kilosort:` settings. Each probe gets its own sort, its own
clock fit and its own output folder; `--probe 1` runs one of them.

Drift correction is off for the Utah array (`blackrock.kilosort.nblocks: 0`):
Kilosort corrects drift by interpolating between neighbouring contacts, and at
400 µm pitch there is no neighbour — a nonzero shift estimate *attenuates* the
signal instead of moving it.

### 1b. In your terminal dir to the SpikeSoritngKilosort dir, Run it

```bash
conda activate kilosort4
python run_sorting_pipeline.py --config configs/session_Porthos_1probe.yaml --machine windows_rig
```

| flag | what it does |
|---|---|
| `--config` | the session YAML (a path, or a name resolved inside `configs/`) |
| `--machine` | profile in `configs/machines/`; defaults to the platform |
| `--dry-run` | report the map, binary and settings; sort nothing |
Don't use dry-run if you want to start real sorting.


This script **sorts and does nothing else** — no sync extraction, no LFP. That is
what lets it go to a cluster (`submit_pipeline.sh` for SLURM) while the rest stays
on the rig. It writes `<system>_dir/kilosort4/`, then stops and prints the exact
`phy` command for each stream it sorted.

---

## Step 2 — curate in Phy

Curation is manual. Run the command Step 1 printed:

On the terminal, cd to the kilorsort4 folder:
```bash
conda activate phy
cd  <neuropixels_dir>/kilosort4/
phy template-gui params.py
```

or directly

```bash
conda activate phy
phy template-gui <neuropixels_dir>/kilosort4/params.py
```

For example, in the above case, it's 
```bash
conda activate phy
cd  Z:/Monkey Porthis/2026-08-13/Porthos_2026_08_13_g0/Porthos_2026_08_13_g0_imec0/kilosort4/
phy template-gui params.py
```


Phy writes `cluster_group.tsv`, whose labels override Kilosort's own
`cluster_KSLabel.tsv`. Everything below reads the human labels where they exist.

## Step 3 — Time alignment and export

```bash
python run_exporting_pipeline.py --config session_Porthos_1probe.yaml --machine windows_rig
```

| stage | what it does |
|---|---|
| `extract_sync` | the 1 Hz train and the 14 s coded burst from each stream. CatGT where the machine has it, always the NumPy detector, and the two compared |
| `lfp` | the LF band, when `export_lfp: true`. Off by default: ~7 GB per hour at 385 channels |
| `time_remapping` | coarse offset from the bursts, fine fit on the 1 Hz train, onto Blackrock time |
| `validate_remapping` | the same map checked against the **held-out** burst onsets; exits non-zero past `alignment_tolerance_s` (1 ms), so it can gate a batch job |
| `export_results` | aligned spike times, metrics, figures |
| `waveforms` | a mean waveform per unit, measured from the binary the sorter read |



### What you get

```
<system>_dir/kilosort4/                 the sorting (Kilosort's own layout)
                                        edge files, nsp_time_map.json, time_map.json
                       export/
                           figures/                     unit_0000.png, overview.png
                           sorting_summary_info.json    what ran, on what, from where
                           <session>_sorted_spikes.mat  times, channel, unit, mean waveform
                           <session>_waveforms.mat      per-spike snippets (when kept)
                           <run>.lfp.mat                the LF band, on Blackrock time
                           units.csv  spike_times.npy  spike_samples.npy
```

One folder per system, and that is the whole tree — handing an analysis a sorting
means handing it a directory. The `.mat` products are MATLAB v7.3, readable by
`load()` in MATLAB and by h5py or `jlab_loader.load_product` in Python;
`sorted_spikes` carries the same fields the online `.nev` container does, so the
same downstream code segments both.

---

##  Installation — install on a new computer

Two conda environments, then this repo. They are separate on purpose: Phy pins an
old Qt stack that fights with Kilosort's, so putting both in one env breaks one of
them. The names below (`kilosort4`, `phy`) are the ones this repo expects — if you
call them something else, say so in the machine profile's `conda_envs:` block.

### 1. The sorting environment — `kilosort4`

```bash
conda create -n kilosort4 python=3.10
conda activate kilosort4
python -m pip install kilosort[gui]
```

Python **3.10 or newer**, because this repo requires it (Kilosort's own docs say
3.9, which is fine for Kilosort alone but too old here). The `[gui]` extra is what
makes `python -m kilosort` — the GUI in Step 0 — work; drop it if you will never
open it.

**Then check the GPU, before anything else.** `torch` arrives as a Kilosort
dependency, and on Windows PyPI serves a **CPU-only** build: it imports fine,
sorts, and silently never touches the GPU.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, replace torch with a CUDA build — pick the command from
<https://pytorch.org/get-started/locally/> for your CUDA version (the lab rig runs
cu128):

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

This repo pins no torch build and installs none — which one you want is a
property of the box, and a pin here would go stale. It only *reports* which one
you ended up with, in 4e.

Kilosort4 install docs: <https://kilosort.readthedocs.io/en/latest/installation.html>

### 2. The curation environment — `phy`

Phy wants its Qt stack from conda rather than pip, so the dependencies go on the
`conda create` line:

```bash
conda create -n phy -y python=3.11 cython dask h5py joblib matplotlib numpy pillow \
    pip pyopengl pyqt pyqtwebengine pytest qtconsole requests responses \
    scikit-learn scipy traitlets
conda activate phy
pip install git+https://github.com/cortex-lab/phy.git
phy --version
```

This environment never runs the pipeline — it only opens `template-gui` on a
finished sorting in Step 2 — so this repo is not installed into it.

Phy install docs: <https://phy.readthedocs.io/en/latest/installation/>

### 3. This repo

```bash
git clone <this repo> && cd SpikeSortingKilosort
conda activate kilosort4
pip install --no-build-isolation -e ".[sorting,dev]"
```

`[sorting]` only declares `kilosort>=4.0`, which 4a already satisfied — installing
Kilosort **first** is what lets you choose the torch build instead of letting a
dependency resolver choose it for you. `[dev]` adds pytest.

Off the rig — laptop, HPC, anything without a GPU — leave both extras off:

```bash
pip install --no-build-isolation -e .
```

That covers every stage except sorting: config, sync extraction, alignment,
export, the tests. It needs no conda env at all, and no Kilosort.

`--no-build-isolation` keeps pip from building in a throwaway env that would
re-resolve numpy and torch behind your back.

### 4. Add a machine profile


`configs/machines/<name>.yaml` holds **only facts about the box** — never data
paths, which belong in the session file. `windows_rig.yaml` and `hpc.yaml` are
tracked; a new laptop needs its own:

```yaml
# configs/machines/mac.yaml
cache_dir: "~/ephys/cache"   # Kilosort's temp.dat -- a LOCAL SSD, never the share
catgt_dir: null
tprime_dir: null
device: "cpu"
```

If you named the two environments something other than `kilosort4` and `phy`, this
is where that is written down:

```yaml
conda_envs:
  sorting: ks5              # shorthand for {name: ks5}
  curation:
    name: phy2
    path: "D:/envs"         # the directory HOLDING the env -> D:/envs/phy2
```

`--machine` defaults to the platform (darwin→`mac`, win32→`windows_rig`, else
`hpc`), overridable with `SPIKESORTING_MACHINE`. Both drivers add `src/` to
`sys.path` themselves, so a checkout runs with no install at all if you prefer.

### 5. Check what you ended up with

```bash
python tools/check_env.py                             # this machine
python tools/check_env.py --machine windows_rig       # another machine's profile
python tools/check_env.py --config configs/<s>.yaml   # also checks the inputs
```

It reports every gap with the stages that gap disables and the exact command that
fixes it — including the CPU-only torch wheel from 4a, which is invisible until
you ask. Read-only, installs nothing, never runs `conda activate` (so it works on
a fresh Windows box whose execution policy still blocks that), and exits non-zero
on anything required, so it can gate a batch job.

### 6. Prove it works

The demo runs the whole pipeline on the official Kilosort4 example recording —
no rig data needed. Run `notebooks/setup_demo_data.ipynb` once (~1 GB recording
plus a channel map), then:

```bash
python run_sorting_pipeline.py   --config configs/session_demo_1probe.yaml
python run_exporting_pipeline.py --config configs/session_demo_1probe.yaml
```

`configs/session_demo_1probe.yaml` is an ordinary session config. Nothing in the
code branches on it being "the demo".

Tests — the pure-compute layer, which needs no GPU, no Kilosort, no
SpikeInterface and no real data:

```bash
python -m pytest tests/ -q
python -m pytest tests/test_align.py -q
python -m pytest tests/ -k "offset or drift" -q
```

### Where things run

| | Windows rig | HPC (Linux) | macOS laptop |
|---|---|---|---|
| Sorting (CUDA) | yes | yes | no |
| CatGT / TPrime | yes | not assumed | no |
| Sync extraction | CatGT + NumPy | NumPy fallback | NumPy fallback |
| Tests, alignment, export | yes | yes | yes |

The stages are independent, so any subset runs on any machine — which is what
lets the GPU half go to a cluster while the CatGT/TPrime half stays on the rig:

```bash
# rig: CatGT and TPrime both live here
python run_exporting_pipeline.py --config <s>.yaml --machine windows_rig --steps extract_sync
# cluster: the GPU work, and the only stage that needs it
python run_sorting_pipeline.py   --config <s>.yaml --machine hpc
# rig: reads the .txt edge files plus spike_times.npy
python run_exporting_pipeline.py --config <s>.yaml --machine windows_rig
```

---

## Extra 1 — probe files (channel maps)

**One rule: the recording's own map wins.** For Neuropixels, `setup_probe` reads
`~snsGeomMap` from the `.meta` beside the binary being sorted, and only falls back
to `neuropixels.probe_file` when the meta predates it. For Blackrock there is no
such file, so `blackrock.probe_file` is **required** — nothing here guesses a Utah
wiring, because real arrays are rarely wired in channel order and a wrong map puts
units on the wrong electrodes with nothing downstream able to tell.

One key per system, whatever the format: `probe_file` is read by extension —
`.cmp` (the array's own wiring map), `.json` (built here) or `.mat` (an older
Kilosort channel map).

```bash
python tools/make_probe.py neuropixels --meta <run>_t0.imec0.ap.meta --out configs/probes/np_<session>.json --plot
python tools/make_probe.py utah --cmp array.cmp --out configs/probes/utah_<array>.json --plot
python tools/make_probe.py from-mat --mat configs/probes/NeuroPix1_default.mat --out configs/probes/np1.json
python tools/make_probe.py show --probe configs/probes/utah_<array>.json --plot
```

**Always pass `--plot` the first time.** Validation catches structural mistakes
(duplicate channels, overlapping contacts, wrong `n_chan`); whether the map matches
how the array is actually wired is something only the picture can tell you.

A relative `probe_file` resolves against the repo, so `configs/probes/x.json`
works from any working directory. `notebooks/build_probe_config.ipynb` builds and
inspects one interactively.

---

## Extra 2 — when a stage skips or fails

Both drivers print one line per stage:

```
[ok] sort_with_kilosort(neuropixels)
       247 units, 1.2 M spikes
[--] extract_lfp(neuropixels)
       export_lfp is false
[!!] validate_remapping(neuropixels)
       ValueError: burst residual 3.1 ms exceeds tolerance
```

`[--]` is not an error — a verb returns `None` when the session says not to run,
and the indented line under it is the reason. Common ones:

| the reason line | what it means |
|---|---|
| `export_lfp is false` | turn it on in the session, or `--lfp-decimate 2` for one run |
| `no blackrock paths declared in the session` | that system has no block, so it did not record |
| `kilosort_on_blackrock is false` | the session says keep the recording, do not sort it |
| `blackrock is the reference timebase; nothing to map it onto` | expected — Blackrock is what everything else is mapped onto |
| `only one system declared, so there is nothing to align against` | no second clock, so alignment and its validation both skip |
| `unknown setting 'export_figure'` | a typo; the error names the closest real key. Every removed key raises rather than being ignored |
| unresolved `{placeholder}` | a `{name}` no longer defined. Only `{blackrock_dir}` and `{neuropixels_dir}` exist |
| neo raises on a timestamp jump | set `blackrock.gap_tolerance_ms`; most real PTP files need it. Above the tolerance the recording splits into segments, and `allow_segments` decides whether that is acceptable |
| both systems share a directory | point `neuropixels_dir` at the `…_imec<n>` folder holding the `.ap.bin` |

A few traps that do *not* announce themselves:

- **`cache_dir` must be a local SSD**, never the network share. Kilosort's
  `temp.dat` is the size of the recording.
- **Two sessions pointed at one directory share every output**, and publishing
  merges rather than clobbers — leaving a blend of two sortings. Point each
  recording at its own folder.
- **Spike times from Kilosort are sample indices**, not seconds, and the real
  sample rate is not the nominal one (29,999.86 Hz measured here — 51 ms of error
  across a 3 h session). `nsp_time_map.json` carries the measured fit; anything
  that divides by 30000 is wrong.

---

## Layout

```
run_sorting_pipeline.py      part 1: sort, and nothing else
run_exporting_pipeline.py    part 2: sync, LFP, alignment, export
submit_pipeline.sh           the same two, under SLURM

configs/                     session_*.yaml (what recorded, and where)
  machines/                  <name>.yaml (facts about the box)
  probes/                    channel maps

src/spikesorting/
  pipeline.py                the verbs, in workflow order -- start here
  _config.py                 SessionConfig, MachineProfile, load_session_config
  _io/  _sync/  _probes/  _export/  _plots/

tools/                       check_env.py, make_probe.py, doctor.py (not stages)
notebooks/                   one verb per cell, same order as the runners
tests/                       pure-compute; no GPU, no sorter, no real data
```

`src/spikesorting/pipeline.py` is the one file to read. Everything underscore-
prefixed below it is machinery.

```python
config = load_session_config(session, machine)   # reads YAML; loads no data

sort_with_kilosort(config, system)               # pipeline 1, and all of it

extract_sync(config, system)      extract_lfp(config, system)
time_remapping(config, system)    validate_remapping(config)
export_results(config, system)    export_waveforms(config, system)
```

Every verb takes `system` (`"neuropixels"` or `"blackrock"`), returns a real value
rather than a status wrapper, and returns `None` when the session says not to run
— which is why both runner scripts are flat lists with no branching.
