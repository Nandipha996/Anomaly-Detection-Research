# Hardware Specification for Research Environment

## Local Machine (Light CPU Environment)

**Device:** MacBook Air (11-inch, Early 2015)  
**Processor:** 1.6 GHz Dual-Core Intel Core i5  
**Cores:** 2 Physical Cores / 4 Threads  
**Memory (RAM):** 4 GB 1600 MHz DDR3  
**Graphics:** Intel HD Graphics 6000 (1536 MB)  
**macOS Version:** macOS Monterey 12.7.6  

**Role:**  
Used only for file organisation, documentation, GitHub sync, and lightweight code checks.  
Primary experiments will *not* run locally due to limited RAM.

---

## Primary Notebook Environment (Web-Based Jupyter)

**Platform:** Jupyter Notebook / JupyterLab in the browser  
**Service:** JupyterHub, Jupyter.org, or a hosted web Jupyter environment  
**Advantages:**  
- No local installation required  
- Stable Python environment  
- Easy to manage notebooks  
- Ideal for baseline experiments and concept-drift simulation  
- Direct GitHub integration for version control  
- Works even with low local RAM  

**Role:**  
This will be your **main environment** for:  
- data exploration  
- baseline models  
- synthetic drift generation  
- experiment organisation

---

## Cloud GPU Environment (for Diffusion Models)

**Platform:** Google Colab (Free Tier)  
**GPU:** NVIDIA T4 (typical assignment)  
**GPU Memory:** ~16 GB  
**Runtime RAM:** ~12.6 GB  
**CPU:** 2 vCPUs (shared)  
**Python:** Python 3.10 (default)  
**Frameworks:** PyTorch (CUDA), Hugging Face Diffusers, Scikit-learn, Numpy, Pandas

**Role:**  
Used **only for diffusion model training and inference**, which benefit from GPU acceleration.

---

## Summary of Environment Strategy

- **Jupyter on the web** → main workspace for coding, analysis, and baselines  
- **Google Colab** → heavy diffusion experiments  
- **MacBook Air** → admin, documentation, GitHub sync, light notebook editing

This structure ensures stability, reproducibility, and no need to switch platforms later.
