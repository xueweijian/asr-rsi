# Verified references

Only statements confirmed from the abstract are listed; numbers not in the abstract are
marked unverified. Do not restate marketing claims as findings.

- 2304.06795 (2023-04) TDT: jointly predicting tokens and durations, skips frames by
  predicted duration.
- 2406.16858 (2024-06) EAGLE-2, 2503.01840 (2025-03) EAGLE-3: dynamic draft trees for
  LLM decoding.
- 2501.14350 (2025-01) FireRedASR: open-source Mandarin ASR with both AED and LLM variants.
- 2505.08699 (2025-05) Granite-speech: speech-aware LLMs.
- 2402.08846 (2024-02) SLAM-ASR: audio adapter into an LLM.
- 2505.13765 (2025-05) WIND: windowed parallel non-blank detection for RNN-T, up to 2.4x
  in greedy modes as reported in the abstract.
- 2406.06220 (2024-06) Label-Looping: efficient transducer decoding.
- 2506.00185 (2025-05) Beam search decoding limits for transducers (ALSD++) — note it is
  a decoding-algorithm paper, NOT a CUDA-graph paper.
- 2206.08317 (2022-06) Paraformer: parallel NAR transformer ASR, CIF-based.
- 2409.17746 (2024-09) Paraformer-v2: noise-robust NAR ASR.
- 2603.11243 (2026-03) CTC-encoder draft self-speculative decoding for LLM-based ASR:
  low frame entropy -> CTC accepted directly, else single-pass verification with relaxed
  criterion; abstract reports WER gains (5.58%, 12%), the speedup factor is NOT verified
  from the abstract.
- 2511.14293 (2025-11) Segmentwise pruning in audio-language models.
- 2603.27981 (2026-03) Encoder-depth pruning of Whisper + LoRA fine-tuning in SLAM-ASR.
- 2510.08599 (2025-10) BaldWhisper: head shearing and layer merging.
- 2602.12241 (2026-02) Moonshine v2 streaming encoder.
- 2608.30927 (2026-08) Stride-k subsampling: Whisper-only, 1500-token interface; stride-2
  keeps WER at baseline in their setup, cuts audio tokens 75% and GFLOPs 52-58%.
- 2604.14493 (2026-04) compact on-device streaming English ASR — NOT the TDT paper.
- 2604.14455 (2026-04) AIBuildAI agent paper.

## Paradigm map

Derived only from the entries above; these are routing priors, not measurements.

| paradigm | entries |
|---|---|
| AED | 2501.14350 (AED variant), 2510.08599, 2602.12241, 2608.30927 |
| SPEECH_LLM | 2501.14350 (LLM variant), 2505.08699, 2402.08846, 2511.14293, 2603.27981 |
| TRANSDUCER | 2304.06795, 2505.13765, 2406.06220, 2506.00185 |
| NAR_CIF_CTC | 2206.08317, 2409.17746, 2603.11243 |
| HYBRID | 2501.14350 (ships both AED and LLM variants) |

Decoding-only LLM work (2406.16858, 2503.01840) and 2604.14455 are not ASR paradigm
evidence; 2604.14493 does not state a paradigm.
