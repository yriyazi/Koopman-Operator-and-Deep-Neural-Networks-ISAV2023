# Leveraging Koopman Operators and Deep Neural Networks for Parameter Estimation and Future Prediction of Duffing Oscillators (ISAV_2023)

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

This repository contains the code and resources related to the paper titled "Leveraging Koopman Operators and Deep Neural Networks for Parameter Estimation and Future Prediction of Duffing Oscillators." In this work, we present a novel approach that combines the power of Koopman operators and deep neural networks to generate a linear representation of the Duffing oscillator. This approach enables effective parameter estimation and accurate prediction of the oscillator's future behavior. We also propose a modified loss function that enhances the training process of the deep neural network. The synergistic use of Koopman operators and deep neural networks not only simplifies the analysis of nonlinear systems but also opens up new avenues for advancing predictive modeling in various scientific and engineering fields.

## Abstract

The study of nonlinear dynamical systems has been fundamental across a range of scientific and engineering domains due to their applicability in modeling real-world phenomena. Traditional methods for analyzing and predicting the behavior of such systems often rely on complex mathematical techniques and numerical simulations. This paper introduces an innovative approach that harnesses the combined potential of Koopman operators and deep neural networks. By generating a linear representation of the Duffing oscillator, we facilitate effective parameter estimation and achieve accurate predictions of its future behavior. Additionally, we propose a modified loss function that refines the training process of the deep neural network. The synergy between Koopman operators and deep neural networks not only simplifies the analysis of nonlinear systems but also holds promise for advancing predictive modeling across diverse fields.

## Contents of the Repository

- `code/`: This directory contains the implementation of the methodology described in the paper. It includes code for generating Koopman operators, training deep neural networks, parameter estimation, and future prediction of Duffing oscillators.

- `data/`: This directory holds the datasets used for training and testing the model. It includes both synthetic data and real-world Duffing oscillator data.

- `results/`: After running the code, the generated results, including parameter estimates, predicted trajectories, and evaluation metrics, will be saved in this directory.

- `notebooks/`: Jupyter notebooks are provided to demonstrate the step-by-step process of using the code for Koopman operator analysis, deep neural network training, and prediction generation.

- `images/`: This directory contains images and plots used in the paper and the README.

- `LICENSE`: The license file associated with this repository.

## Getting Started

To get started with using the code and reproducing the results presented in the paper, please refer to the `notebooks/` directory. The Jupyter notebooks provide a clear guide on how to set up the environment, preprocess data, execute code, and interpret the results.

## Citation

If you find this work helpful or build upon it in your research, please consider citing the following paper:

```
[Author(s). (Year). Title of the Paper. Journal/Conference Name, Volume(Issue), Page Numbers. DOI]
```
<p float="center">
<img src="Images\alpha = 0000 gamma=0.37.png" alt="GIF Example">
</p>


## Contact

For any inquiries, issues, or collaboration opportunities, please contact [author@email.com].

We hope that the approach introduced in this paper will inspire further advancements in the analysis and prediction of nonlinear dynamical systems. Happy researching!
