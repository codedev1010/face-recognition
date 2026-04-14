# Asymmetric Hybrid-NCRR for Robust Facial Recognition


This repository contains the Jupyter Notebook implementations of the **Asymmetric Hybrid Non-Convex Relaxation Regularized Regression (Hybrid-NCRR)** framework. This algorithm is engineered to perform highly accurate facial recognition under severe, unconstrained environmental degradation.

## 🧠 The Asymmetric Hybrid Methodology

Traditional Sparse Representation and Collaborative Representation classifiers often fail when subjected to mixed domains of noise because they blindly apply a single penalty function to all error types. 

This model introduces a novel **Asymmetric** objective function solved via the Alternating Direction Method of Multipliers (ADMM) framework, treating different noise topologies with distinct mathematical bounds:

* **The "Leash" (Logarithmic SVD Shrinkage):** Applied to the low-rank error matrix ($L$). Its slow, linear gradient decay acts as a leash, safely bounding massive contiguous block occlusions without causing singular value leakage from the underlying facial structure.
* **The "Scalpel" (Arctangent Element-wise Shrinkage):** Applied to the sparse error matrix ($S$) and group sparsity vectors ($v$). Its rapid, quadratic gradient decay combined with a hard Newton-Raphson threshold surgically isolates independent pixel static without eroding faint, authentic facial features.

## 🔬 Tested Noise Domains

The optimization loop has been empirically tested and proven robust against three distinct noise scenarios, with dedicated code available for each:

1. **Contiguous Block Occlusion (`black box occulusion(new penalty function).ipynb`):** Simulating targeted spatial blockages (e.g., scarves, hands, or dense shadows) using Generalized Singular Value Thresholding.
2. **Sparse Pixel Corruption (`pixel noise(new penalty function).ipynb`):** Simulating heavy, independent sensor static and Gaussian noise (handling corruption levels up to 90%).
3. **Mixed Noise (`mixedNoise(new penalty function).ipynb`):** The ultimate unconstrained environment, successfully recovering facial identities when subjects are simultaneously obscured by dense blocks and heavy static.

## ⚙️ Repository Structure

* `CV codes/`: Contains the primary Jupyter Notebooks implementing the mathematical framework for the different noise scenarios:
    * `black box occulusion(new penalty function).ipynb`
    * `mixedNoise(new penalty function).ipynb`
    * `pixel noise(new penalty function).ipynb`
* `cropped/` : Directory for the dataset used in the work

## 🚀 Getting Started

### Prerequisites
The core mathematics rely on standard, highly-optimized Python scientific libraries. No heavy GPU frameworks are required.
* Python 3.8+
* `numpy`
* `scipy`
* `jupyter`

### Installation & Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/codedev1010/face-recognition.git](https://github.com/codedev1010/face-recognition.git)
   cd face-recognition
2. Navigate to the code directory:
    ```bash
   cd "CV codes"
3. Launch the Jupyter environment:
    ```bash
    jupyter notebook
4. Open the desired .ipynb file based on the noise scenario you want to test and run the cells sequentially to observe the ADMM optimization separating the test image into the clean dictionary reconstruction ($Az$), the block occlusion ($L$), and the pixel noise ($S$).
   
   
