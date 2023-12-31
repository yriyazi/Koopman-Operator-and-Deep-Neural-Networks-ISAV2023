# Leveraging Koopman Operators and Deep Neural Networks for Parameter Estimation and Future Prediction of Duffing Oscillators (ISAV_2023)
<p> Updated</p>
<!-->
<p float="center">
  <img src="Images\Duffing Oscillator (δ=0.3, α=-1.0, β=1, γ=0.2, ω=1.2).png"  />
</p>
<-->

<p float="center">
  <img src="Images\Duffing Oscillator (δ=0.3, α=-1.0, β=1, γ=0.29, ω=1.2).png"  />
</p>

<p float="center">
  <img src="Images\Duffing Oscillator (δ=0.3, α=-1.0, β=1, γ=0.37, ω=1.2).png"  />
</p>

Duffing Solutions

![Example GIF](Images\3d_phase_space_animation.gif)
<p float="center">
<img src="Images\3d_phase_space_animation.gif" alt="GIF Example">
</p>

This repository contains the code and resources related to the paper titled "Leveraging Koopman Operators and Deep Neural Networks for Parameter Estimation and Future Prediction of Duffing Oscillators." In this work, we present a novel approach that combines the power of Koopman operators and deep neural networks to generate a linear representation of the Duffing oscillator. This approach enables effective parameter estimation and accurate prediction of the oscillator's future behavior. We also propose a modified loss function that enhances the training process of the deep neural network. The synergistic use of Koopman operators and deep neural networks simplifies the analysis of nonlinear systems. It opens new avenues for advancing predictive modeling in various scientific and engineering fields.

## Abstract

The study of nonlinear dynamical systems has been fundamental across various scientific and engineering domains due to their applicability in modeling real-world phenomena. Traditional methods for analyzing and predicting the behavior of such systems often rely on complex mathematical techniques and numerical simulations. This paper introduces an innovative approach that harnesses the combined potential of Koopman operators and deep neural networks. By generating a linear representation of the Duffing oscillator, we facilitate effective parameter estimation and achieve accurate predictions of its future behavior. Additionally, we propose a modified loss function that refines the training process of the deep neural network. The synergy between Koopman operators and deep neural networks simplifies the analysis of nonlinear systems and holds promise for advancing predictive modeling across diverse fields.

## Contents of the Repository

- `code/`: This directory contains the implementation of the methodology described in the paper. It includes code for generating Koopman operators, training deep neural networks, parameter estimation, and future prediction of Duffing oscillators.

- `data/`: This directory holds the datasets used for training and testing the model. It includes both synthetic data and real-world Duffing oscillator data.

- `results/`: After running the code, the generated results, including parameter estimates, predicted trajectories, and evaluation metrics, will be saved in this directory.

- `notebooks/`: Jupyter notebooks are provided to demonstrate the step-by-step process of using the code for Koopman operator analysis, deep neural network training, and prediction generation.

- `images/`: This directory contains images and plots used in the paper and the README.

- `LICENSE`: The license file associated with this repository.

## Getting Started

To start using the code and reproducing the results presented in the paper, please refer to the `notebooks/` directory. The Jupyter notebooks provide a clear guide on how to set up the environment, preprocess data, execute code, and interpret the results.

<!-- ![Example GIF](Images\3d_phase_space_animation.gif) -->
<p float="center">
<img src="Duffing_Solution\results\Poncare section\Poincaré Map of the Duffing OscillatorFrames=600 points=800 All=True gamma=0.37 omega=1.2 beta=1 alpha=-1.0 delta=0.3.gif" alt="GIF Example">
</p>



## Citation

If you find this work helpful or build upon it in your research, please consider citing the following paper:

```
[Yassin Riyazi, Navidreza Ghanbari, Arash Bahrami*. 2023. "Leveraging Koopman Operators and Deep Neural Networks for Parameter Estimation and Future Prediction of Duffing Oscillators." ISAV, 2023, Page Numbers. DOI]
```
<p float="center">
<img src="Images\alpha = 0000 gamma=0.37.png" alt="GIF Example">
</p>


## Contact

If you have any questions, issues, or collaboration opportunities, please contact [iyasiniyasin98@gmail.com].

We hope the approach introduced in this paper will inspire further advancements in analyzing and predicting nonlinear dynamical systems. Happy researching!
