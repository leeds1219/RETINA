# ✨ RETINA: Relational Entity Text-Image kNowledge Augmented Benchmark

This repository provides the **RETINA** benchmark, a novel and large-scale dataset for **Multimodal Knowledge-Based Visual Question Answering (MKB-VQA)**.

RETINA was introduced to overcome a critical limitation in existing MKB-VQA datasets: the **"visual shortcut."** Previous models could often succeed by simply matching the query image to the target document's primary subject entity.

## 🚀 Key Feature: Breaking the Shortcut

RETINA is explicitly designed to eliminate this bias, forcing models to rely on true relational knowledge and multi-hop reasoning:

The benchmark's construction process ensures that the **query image** is of a **secondary, related entity** mentioned in the document, rather than the main subject. This setup reflects complex, real-world scenarios where knowledge retrieval must go beyond direct visual matching.

## 💾 Dataset Access

The full dataset, including the large training set and the human-curated test set, is available for download and use on Hugging Face:

[**Access the RETINA Dataset**](https://huggingface.co/datasets/Lee1219/RETINA)

### Statistics

| Component | Size | Note |
| :--- | :--- | :--- |
| **Training Set** | 120k samples | Automatically generated via an LLM-driven pipeline. |
| **Test Set** | 2k samples | Human-curated. |

## ⚖️ Research Use and Liability Disclaimer

The RETINA dataset is intended for **non-commercial research purposes**. Users are solely responsible for any and all utilization of the dataset. The creators of this benchmark and their affiliated institutions shall not be held liable for any damages, consequences, or legal issues that may arise from its use.
