<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app/host/public/brand/strata-lockup-dark.svg">
    <img src="app/host/public/brand/strata-lockup-light.svg" alt="Strata-Net" width="340">
  </picture>
</p>

<h1 align="center">Strata-Net</h1>

<p align="center"><sub>Strata-Net's mark uses a cyan middle bar so it is not mistaken for upstream Strata's.</sub></p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#one-model-on-two-or-more-pcs">Two or more PCs</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="docs/lang/README.zh-CN.md">简体中文</a> · <a href="docs/lang/README.ja.md">日本語</a> · <a href="docs/lang/README.de.md">Deutsch</a> · <a href="docs/lang/README.fr.md">Français</a> · <a href="docs/lang/README.es.md">Español</a> · <a href="docs/lang/README.pt-BR.md">Português</a> · <a href="docs/lang/README.pl.md">Polski</a></sub></p>

**Strata-Net** is a fork of [Strata](https://github.com/Niko1221/Strata) that brings the work of two other forks into
one project: lask3802's [Strata-Link](https://github.com/lask3802/Strata-Link), which splits one model across PCs on a
LAN, and fragtion's [StrataPool](https://github.com/fragtion/StrataPool), whose parts are being added one at a time.
The goal is to run a model on hardware people already have by pooling their PCs, including PCs that are not on the
same network. It is mainly a feasibility project: the first question is whether a split model is usable over home
internet. Today the split runs on a LAN; running it over the internet is planned and not built
([what works today](#one-model-on-two-or-more-pcs)). Upstream Strata releases keep being merged in.

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

<a id="strata-link-one-model-on-two-or-more-pcs"></a>

## One model on two or more PCs

Strata-Net can split one model across PCs. The split is lask3802's remote stage from Strata-Link, and today it runs
between PCs on the same network, over TLS 1.3 with each end checking the other's certificate. This helps when the
model does not fit one PC well, but you have a second PC with a graphics card.

- **Each PC loads only its own layers' experts into RAM**, so no PC needs RAM for the whole model. Every PC still needs
  the model's files; `terminal/tools/stage_ship.py` sends a worker only its layers.
- **Every card adds memory for the expert cache.** More of the experts are on a GPU, so fewer come from RAM.
- **Decoding can get faster, or slower.** In lask3802's measurements below, two PCs decode 16.8% faster than the
  2080 Ti alone. A slow worker or link can make it slower: a 3080 with a 3070 under WSL2 over 1 GbE decoded short
  prompts at 36.2 tok/s, against 39.9 for the 3080 alone ([measurements](docs/REMOTE_STAGE.md#when-it-helps-measured)).
- **The split is opt-in.** It runs only if you start it with `--remote-stage` on the main PC and `--stage-worker` on
  the others. One Link change applies on one PC too: on NVIDIA, a layer's CPU experts join one layer later (+15.2%
  decode on an RTX 3080, the answers differ slightly; `STRATA_VERIFY_DEFER_CPU=0` turns it off,
  [details](docs/DETAILS.md)).

Measured by lask3802 on 2026-10-23 with Link's build of engine 0.1.48 (upstream 0.2.3; it was not released on its own): an RTX 2080 Ti 11 GB, plus an RTX 3080 10 GB in a second PC over
10 GbE (about 400 MB/s, limited by its slot). These are PCs A and B of
[docs/REMOTE_STAGE.md](docs/REMOTE_STAGE.md#when-it-helps-measured) with their cards swapped. The model is SC117's
GSQ-RCO abliterated IQ3_S of Qwen3.8-Flash-Next; on two PCs, layers 0-19 run on the 2080 Ti. Settings as served:
262,144 context, `--kv int8`, `--spec 4` with the draft layer, `--prefill auto` alone and 5,888 on two PCs. Each number is the median of three
rounds in one session (12 short prompts a round; three prompts of 29,000-29,800 tokens). A second two-PC session gave
70.28 and 1,382 / 63.0. On Link's build of engine 0.1.49.1 (upstream 0.2.4.1, 2026-10-24), the same two PCs, split,
model and context gave 70.22 and 1,269 / 63.6 (one-PC rows not run again).

| | 2080 Ti alone | 2080 Ti + 3080 (two PCs) |
| --- | ---: | ---: |
| short prompts, decode tok/s | 60.31 | 70.42 (+16.8%) |
| ~30K prompt, prefill / decode tok/s | 833 / 49.4 | 1,291 / 63.7 |

What has been run, and what has not:

| | |
| --- | --- |
| **Measured by Strata-Link** (lask3802) | NVIDIA cards only (RTX 2080 Ti, 3080, 3070), on two and three PCs; a Linux main PC; Linux, Windows 11 and WSL2 workers; 1 GbE and 10 GbE links; the Qwen3.8-Flash-Next family only |
| **Measured by Strata-Net** | Two Linux PCs, an RTX 3080 main and an RTX 5070 worker, over 1 GbE |
| **Reported by a user** | An RTX 5070 main PC with a Radeon 8060S worker (AMD, HIP), both on Windows, over USB4; SC117's IQ3_XXS; on 0.1.45.3-link.1 ([report](https://github.com/Niko1221/Strata/pull/1748#issuecomment-6332714185)) |
| **Not run by either** | An AMD main PC, a Windows main PC (one user report above), other models |
| **Not supported** | Intel cards (the SYCL engine has no remote stage) and a Mac (the Metal engine refuses it at start) |

What works today, in this tree:

- **The split on a LAN**, as described above.
- **A slow link.** The main engine's watchdog does not count the time a send or receive on the link is under way, up
  to the link's own timeout, so a long prompt can finish over a slow link and a dead link still ends the request. Each
  row stream on the link has its own format, and the rows only the drafter reads come back in bf16, half the bytes
  ([details](docs/REMOTE_STAGE.md#options)).
- **A PC or the link dropping mid-answer.** The request waits, the web chat shows "system dropped - recovering...",
  and the answer carries on once the link is back. The main engine stays loaded through a lost link, and after a
  reconnect to the same worker process it resumes from the last checkpoint both sides hold
  ([details](docs/REMOTE_STAGE.md#a-pc-drops-in-the-middle-of-an-answer)).
- **Real events while a long prompt is read**, every 5 s, so a client that only counts events does not take a slow
  read for a dead server ([details](docs/DETAILS.md#streaming-and-tool-calls)).
- **fragtion's split planner**, as a library, with link speed and delay added to its costs. It is built and tested,
  but nothing calls it yet.

Slow links have only been tested on a LAN with two PCs (an RTX 3080 and an RTX 5070), through a proxy that slows the
link to 15 ms and 20 Mbit/s each way. No run between two homes has happened yet.

Planned, not built:

- Joining a pool over the internet by invite, into a pool that an admin runs. The first proof of concept uses
  Tailscale.
- A pool page where the admin picks a model and splits it across the pool's PCs, with the planner placing the layers.
  Several admins per pool, with roles.
- Each member controls their own PC: they can cap what it shares and pause it.
- Windows and Linux with NVIDIA cards first, and one request at a time.
- A security review of everything new.

Setup, the earlier measurements and the limits: [docs/REMOTE_STAGE.md](docs/REMOTE_STAGE.md). This fork's releases:
[ZaneZaiontz/Strata-Net](https://github.com/ZaneZaiontz/Strata-Net/releases).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.16-windows-x64-setup.exe` · `Strata-0.2.16-linux-x86_64.AppImage` · `Strata-0.2.16-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

Two or three cards of the same make in one PC can share a model ([multi-GPU](docs/MULTI_GPU.md)); across PCs, see
[two or more PCs](#one-model-on-two-or-more-pcs).

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

## Credits

- **Strata** is by [Niko1221](https://github.com/Niko1221) and the Strata contributors:
  [Niko1221/Strata](https://github.com/Niko1221/Strata). Strata-Net is built on it. Model and library credits:
  [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
- **The remote stage** is by [lask3802](https://github.com/lask3802): the split across PCs and everything built on it,
  including its TLS, the node agent, `stage_ship`, the tuner, checkpoints, parking and session files. It was proposed
  upstream as [PR #1748](https://github.com/Niko1221/Strata/pull/1748), and this fork comes from
  [lask3802/Strata-Link](https://github.com/lask3802/Strata-Link).
- **From [StrataPool](https://github.com/fragtion/StrataPool)**, by [fragtion](https://github.com/fragtion): the
  layer-major prompt-reading fix for 256-expert models; the split planner (`engine/src/net/stage_split.cpp`), which
  picks where to cut the model from each PC's free memory and GPU speed; and the Windows build fixes (`/bigobj` for
  MSVC, and the compiler probe without debug records for the engine and the image generator). A row format for each
  stream on the link is StrataPool's idea too; Strata-Net built its own on Link's protocol. More of StrataPool is
  being added one part at a time, and each part is credited here when it lands.
- **The real events while a long prompt is read** are by [suifei](https://github.com/suifei), from upstream
  [PR #4340](https://github.com/Niko1221/Strata/pull/4340) (closed unmerged), ported by way of StrataPool.
- [YukihiroHemmi](https://github.com/YukihiroHemmi) (Arinko) read the link's code while building a SYCL worker, and
  [their report](https://github.com/Niko1221/Strata/pull/1748#issuecomment-6138154786) led to the frame-size hardening
  ([details](docs/remote-stage/ENGINEERING.md#a-readers-report-what-a-peer-can-make-a-process-do-2026-10-13)).
- [edowadolui-lab](https://github.com/edowadolui-lab) ran a Radeon 8060S worker with an RTX 5070 main PC and
  [reported it](https://github.com/Niko1221/Strata/pull/1748#issuecomment-6332714185).
- [ZaneZaiontz](https://github.com/ZaneZaiontz) maintains this fork and merges upstream releases into it, and wrote
  the slow-link and recovery changes and the planner's link costs.

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](.github/CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (one line per public supporter, newest last; copy a line and change the GitHub name in all four places) -->
<a href="https://github.com/ryan4yin"><img src="https://github.com/ryan4yin.png?size=96" width="48" height="48" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://github.com/themactep.png?size=96" width="48" height="48" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://github.com/macintog.png?size=96" width="48" height="48" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://github.com/bisonbet.png?size=96" width="48" height="48" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://github.com/vizv.png?size=96" width="48" height="48" alt="vizv" title="vizv"></a>
<a href="https://github.com/kod88vn"><img src="https://github.com/kod88vn.png?size=96" width="48" height="48" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/seoulworker"><img src="https://github.com/seoulworker.png?size=96" width="48" height="48" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jackyini"><img src="https://github.com/jackyini.png?size=96" width="48" height="48" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/router0mail"><img src="https://github.com/router0mail.png?size=96" width="48" height="48" alt="router0mail" title="router0mail"></a>
<!-- supporters:end -->

All supporters and how to support Strata: [SUPPORTERS.md](SUPPORTERS.md)

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

