# Multimodal LESS: Selecting Influential Data for Targeted Instruction Tuning

## Abstract
This project implements the **Multimodal LESS** (Low-rank gradient Similarity Search) framework to address the challenge of data selection in Vision-Language Models (VLMs). By selecting high-quality training data from `LLaVA-Instruct-150K` that specifically improves performance on targeted skills, we demonstrate a compute-efficient method for targeted instruction tuning.

## Methodology
The framework operates in five stages:
1.  **Model Setup:** Leveraging `llava-hf/llava-1.5-7b-hf` (Pre-trained).
2.  **Gradient Extraction:** Computing gradients for specific modules (e.g., Multimodal Projector) for both the Training Pool and a Target Validation Set.
3.  **Projection:** Using Random Projections to compress high-dimensional gradients into low-rank features.
4.  **Selection:** Calculating Cosine Similarity between Target Gradients and Training Gradients to select the top 5% most influential samples.
5.  **Fine-tuning & Evaluation:** Training on the selected subset and benchmarking against random baselines.

## Repository Structure
* `final_project_submission.ipynb`: The complete end-to-end implementation code, including data preparation, gradient extraction, selection algorithm, and final visualization of results.

## Key Results
* **Information Gain:** The selected subset showed a higher information density compared to random sampling.
* **Qualitative Improvement:** The model fine-tuned on the LESS-selected subset demonstrated significantly better adherence to complex reasoning instructions compared to the baseline.
* **Steerability:** The algorithm successfully identified data semantically aligned with the target reasoning tasks.

## Requirements
* Python 3.10+
* PyTorch
* Transformers
* PEFT (Parameter-Efficient Fine-Tuning)
* BitsAndBytes
* Scikit-learn
* Matplotlib / Seaborn

## Usage
Open `final_project_submission.ipynb` in Jupyter Lab or Google Colab and run the cells sequentially. Ensure GPU acceleration is enabled.
