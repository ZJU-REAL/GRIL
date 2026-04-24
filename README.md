<h1 align="center">
Pause or Fabricate? Training Language Models for Grounded Reasoning
</h1>
<div align='center' style="font-size:18px;">
<p>
    <a href="https://arxiv.org/abs/2604.19656">
      <img src="https://img.shields.io/badge/Paper-arxiv-blue" alt="Paper"/>
    </a>
</p>
</div>
<p align="center">
Yiwen Qiu, Linjuan Wu, Yizhou Liu, Yuchen Yan, Xu Tan, Jin Ma, Yao Hu,
Daoxin Zhang, Wenqi Zhang, Weiming Lu, Jun Xiao, Yongliang Shen*
</p>
<p align="center">
Zhejiang University &nbsp;|&nbsp; Xiaohongshu Inc. &nbsp;|&nbsp; Tencent
</p>
---
 
## 🔥 Overview
 
We propose **GRIL** (**G**rounded **R**easoning via **I**nteractive **R**einforcement **L**earning), a multi-turn RL framework that trains language models to recognize **inferential boundaries** and reason only when sufficient information is available.
 
A key failure mode in current LLMs — which we term **ungrounded reasoning** — is the tendency to fabricate missing premises and proceed with elaborate but baseless inference chains, rather than pausing to request clarification.
 
<!-- [Overview Figure Placeholder] -->
 
GRIL decomposes reasoning into two stages:
1. **Clarify and Pause**: the model determines whether inputs contain sufficient premises, earning a time-decayed reward for early detection of missing information.
2. **Grounded Reasoning**: once missing premises are supplied by the environment, the model integrates them to produce a valid solution.
<!-- [Method Figure Placeholder] -->
 
---
 
## 📊 Main Results
 
GRIL achieves substantial improvements across model scales on **GSM8K-Insufficient** and **MetaMATH-Insufficient** benchmarks.
 
| Model | SR ↑ | PD ↑ | NT ↓ | Length ↓ |
|---|---|---|---|---|
| Qwen2.5-1.5B Base | 1.8 | 4.6 | 3.828 | 810 |
| Qwen2.5-1.5B + GRIL | **61.6** | **90.8** | **2.913** | **479** |
| Qwen2.5-3B Base | 20.6 | 28.0 | 3.665 | 887 |
| Qwen2.5-3B + GRIL | **73.5** | **88.0** | **2.481** | **448** |
| Qwen3-1.7B Base | 41.3 | 52.6 | 3.271 | 1271 |
| Qwen3-1.7B + GRIL | **72.8** | **96.5** | **2.348** | **376** |
 
SR: Success Rate (%), PD: Premise Detection (%), NT: Avg. Interaction Turns, Length: Response Length in tokens. Results on GSM8K-Insufficient.
 
Notably, GRIL **does not degrade** performance on complete problems — it also improves standard GSM8K (e.g., Qwen2.5-1.5B: 52.0% → 69.7%) and MATH500 results.
 
<!-- [Results Table / Figure Placeholder] -->
 
---
 
## 🗞️ News
 
- **`2026-04`**: Paper released on arXiv.
---
 
## 🛠️ Installation
 
Code is coming soon.
 
---
 
## 📐 Method
 
### Problem Formulation
 
We formalize reasoning under incomplete information as a Markov Decision Process **M = ⟨S, A, P, R, γ⟩**, with two abstract inferential actions:
- **Solve** (*a*<sub>solve</sub>): commit to current information as sufficient and produce a solution.
- **Clarify** (*a*<sub>clarify</sub>): identify missing premises and explicitly request clarification.
Ungrounded reasoning occurs precisely when models select *a*<sub>solve</sub> in states where *a*<sub>clarify</sub> is appropriate.
 
### Stage-Specific Rewards
 
**Stage 1 — Clarify and Pause** (incomplete problems):
 
$$R_{\text{detect}} = r_{\text{base}} \cdot \gamma_d^{n_{\text{prior}}}$$
 
A time-decay factor γ<sub>d</sub> ∈ (0, 1) rewards *early* detection, directly penalizing the "early suspicion, late action" pattern common in base models.
 
**Stage 2 — Grounded Reasoning** (after premise provision):
 
$$R_{\text{solve}} = r_{\text{correct}} \cdot \mathbf{1}[\text{answer is correct}]$$
 
**Complete Problem Handling**:
 
$$R_{\text{comp}} = r_{\text{correct}} \cdot \mathbf{1}[\text{correct}] - \lambda \cdot \mathbf{1}[\text{unc.}]$$
 
A penalty λ discourages trivially requesting clarification on well-formed inputs.
 
---
 
## 🔬 Analysis Highlights
 
- **GapRatio reduction**: GRIL substantially lowers the proportion of tokens generated after initial uncertainty signals (e.g., SVAMP: 0.63 → 0.27).
- **Precise discrimination**: F1 scores of 0.908 and 0.931 on mixed complete/incomplete input sets, with Precision above 0.96.
- **Out-of-distribution generalization**: Consistent improvements on HotpotQA-Insufficient and CommonsenseQA-Insufficient without domain-specific training.
- **Robustness to noisy interactions**: Under noisy feedback, Qwen2.5-1.5B improves from 12.8% to 47.2% Success Rate.
---
 
## ⭐️ Citation
 
If you find this work useful, please consider citing:
 
```bibtex
@misc{qiu2026pausefabricatetraininglanguage,
      title={Pause or Fabricate? Training Language Models for Grounded Reasoning}, 
      author={Yiwen Qiu and Linjuan Wu and Yizhou Liu and Yuchen Yan and Jin Ma and Xu Tan and Yao Hu and Daoxin Zhang and Wenqi Zhang and Weiming Lu and Jun Xiao and Yongliang Shen},
      year={2026},
      eprint={2604.19656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.19656}, 
}
```
 
---
 
## 🤝 Acknowledgement
 
This work was supported by the National Natural Science Foundation of China (No. 62506332, No. 62436007), CCF-Tencent Rhino-Bird Open Research Fund, and ZJU Kunpeng&Ascend Center of Excellence.
 
This project builds on [veRL](https://github.com/volcengine/verl). We thank the authors of that project.
 
