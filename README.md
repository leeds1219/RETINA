# ✨ RETINA: Relational Entity Text-Image kNowledge Augmented Benchmark

This repository provides the **RETINA** benchmark, a novel and large-scale dataset for **Multimodal Knowledge-Based Visual Question Answering (MKB-VQA)**.

RETINA was introduced to overcome a critical limitation in existing MKB-VQA datasets: the **"visual shortcut."** Previous models could often succeed by simply matching the query image to the target document's primary subject entity.

## 🚀 Key Feature: Breaking the Shortcut

RETINA is explicitly designed to eliminate this bias, forcing models to rely on true relational knowledge and multi-hop reasoning:

The benchmark's construction process ensures that the **query image** is of a **secondary, related entity** mentioned in the document, rather than the main subject. This setup reflects complex, real-world scenarios where knowledge retrieval must go beyond direct visual matching.

## 💾 Dataset Access

The RETINA bench, including the large training set and the human-curated test set, is available for download and use on Hugging Face:

[**Access the RETINA Dataset**](https://huggingface.co/datasets/Lee1219/RETINA)

For EVQA and Infoseek, including the query images and textual KB, please refer to [Lin Weizhe et al.](https://arxiv.org/abs/2402.08327):

[**Access the M2KR Dataset**](https://github.com/LinWeizheDragon/FLMR/tree/main)

For the document images please refer to [Lianghao Deng et al.](https://github.com/lhdeng-gh/MuKA).

### Statistics

| Component | Size | Note |
| :--- | :--- | :--- |
| **Training Set** | 120k samples | Automatically generated via an LLM-driven pipeline. |
| **Test Set** | 2k samples | Human-curated. |

### Multi-hop Extension
**(Status: TBD)**

~~We are also preparing an additional multi-hop version of the dataset built on top of the original one-hop setting.~~

~~Since multi-hop examples are much harder to curate, this version is currently being generated without full manual curated test set.~~

**This multi-hop dataset is independent of the paper and is provided solely as a community resource.**

## Acknowledgements
We build apon [LinWeizheDragon/FLMR](https://github.com/LinWeizheDragon/FLMR/tree/main) and [lhdeng-gh/MuKA](https://github.com/lhdeng-gh/MuKA)

For image verification we use [Imagehash toolkit](https://github.com/jenssegers/imagehash).

## ⚖️ Research Use and Liability Disclaimer

The RETINA dataset is intended for **non-commercial research purposes**. Users are solely responsible for any and all utilization of the dataset. The creators of this benchmark and their affiliated institutions shall not be held liable for any damages, consequences, or legal issues that may arise from its use.
