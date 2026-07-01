# Overview

This is a testing sandbox for developing various methods of injecting symbolic
knowledge into neural networks. This symbolic NN approach is intended to
facilitate the training of models for physics based problems where process
complexity and data availability are generally significant hindrances to the
application of NNs. By injecting known, symbolic human knowledge we aim to guide
the model towards the correct embedding space more quickly than could be
achieved through pure statistical loss.

# Experiments

## Experiment 1 - Relationally embedded Semantic Loss Functions (SLF)

### Planning

- Simple semantic loss based on intuitively noticeable properties of numeric
  characters
- Makes use of manually made reward matrices to weight rewards depending on how
  "close" the model was
- Very rudimentary example of semantic loss

#### Mathematical Implementation Details

Initially, the semantic penalty was multiplied by the cross entropy loss and
summed over the batch. This unintentionally acted as a massive scalar multiplier
on the gradients (by a factor of the batch size). The semantic maths have since
been refined to: 1. **Probabilities:** Apply `softmax` to raw logits to convert
them to probabilities before absolute difference calculations. 2. **Scale
Invariance:** Average (`.mean()`) the penalty across the batch instead of
summing (`.sum()`) to prevent batch size from inflating the penalty magnitude.
3. **Additive Loss:** Add the penalty to the base loss
(`ce_loss + alpha * semantic_penalty`) rather than multiplying them, preventing
extreme, unintentional learning rate inflation on the base gradients.

### Results

- Training loss ![Training loss plot for experiment
  1](./results/Experiment1/train_loss.png)
- Validation loss ![Validation loss plot for experiment
  1](./results/Experiment1/val_loss.png)
- Test loss ![Test loss plot for experiment
  1](./results/Experiment1/test_loss.png)

### Conclusions

- Seems to have worked
  - Clear improvement in training rate with semantics added
  - Similarity cross entropy in particular shows clear signs in validation loss
    of being on a similar complementary CDF to the normal cross-entropy loss,
    but training faster
- Interestingly: the "garbage" cross entropy seems to have also produced a very
  good result! This is likely because it hasn't been normalized to 1, so it may
  be simply amplifying the gradient by random amounts at all times. Basically
  acting as a fuzzy gradient booster.
- I would consider this experiment a success, with some interesting open
  questions remaining worth further examination

## Experiment 2 - Dataset qualitative characteristic derived SLFs

### Planning

- Makes use of known physics equations that partially describe the problem to
  guide the model
  - Reduces the need for the model to learn known physics, allowing it to focus
    on learning the unknown physics
  - Should accelerate training
  - Allows for easier model minimisation
- Dataset selection will be important here
  - Needs to be a familiar area to the experimenter
  - Needs to be openly available
  - Needs to describe a problem complex enough that we wouldn't just be fitting
    on a known equation (e.g: if we were to try Hooks law we would just fit to
    the equation which would be pointless)
- Possible candidate datasets:
  - [Molecular
    Properties](https://www.kaggle.com/datasets/burakhmmtgl/predict-molecular-properties)
  - [Nuclear Binding
    Energy](https://www.kaggle.com/datasets/iitm21f1003401/nuclear-binding-energy)
  - [Body Fat
    Prediction](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)
- Decided to use Molecular Properties dataset as it is quite familiar to me
- Training with semantics added to renationship between molecular energy and
  differential electronegativity
  - Semantics being injected are:
    - These values should be positively corellated
    - These values should be weighted towards a high r^2 with an adaptive
      penalty
  - Multiple attempts carried out:
    - Simple penalties. Variations tested include:
      ```math
      Loss = ( Softplus( -m ) + 1 ) * SmoothL1Loss
      ```
      ```math
      Loss = ( Relu( -m ) + 1 ) * SmoothL1Loss
      ```
      ```math
      Loss = ( \frac{1}{Sech(|r|)} + 1 ) * SmoothL1Loss
      ```
      ```math
      Loss = ( {r}^2 + 1) * SmoothL1Loss
      ```

    - Adaptive, self training penalties tuned by various methods. Best method
      found was optimisation by a random forest regressor. These tunable
      variants include:
      ```math
      Loss = ( Softplus( \alpha * -m ) + 1 ) * SmoothL1Loss
      ```
      ```math
      Loss = ( Relu( \alpha * -m ) + 1 ) * SmoothL1Loss
      ```
      ```math
      Loss = ( \frac{ 1 }{ Sech( \alpha * |r| ) } + 1 ) * SmoothL1Loss
      ```
      ```math
      Loss = ( \alpha * { r }^2 + 1) * SmoothL1Loss
      ```

    - Final adaptive semantic loss function tested (called
      `positive_slope_linear_loss`) was the following:
      ```math
      Loss = ( \alpha * { r }^2 + 1) * ( \frac{ 1 }{ \beta } * log( 1 + exp( \beta * \gamma * -m ) ) + 1 ) * SmoothL1Loss
      ```

### Results

- Training loss ![Training loss plot for experiment
  2](./results/Experiment2/train_loss.png)
- Validation loss ![Validation loss plot for experiment
  2](./results/Experiment2/val_loss.png)
- Test loss ![Test loss plot for experiment
  2](./results/Experiment2/test_loss.png)

### Conclusions

- Method didn't appear to work too well because:
  - Simple loss functions tested were likely suboptimal for effectively
    influencing model
  - Guesses at parameters in simple functions need to be optimised, basically
    turning this into a hyperparameter optimisation problem, which defeats the
    purpose of semantic loss
  - Adaptive, ML based loss functions do not appear to be converging quickly
    enough to train the model faster than the normal loss functions
- For this reason, I would conclude this experiment as a failure

## Experiment 3 - Tensor Logic Network (TLN)

- Revisiting QMNIST from Expt1 but using TLN instead of SLF.
- Using LTNtorch library or maybe Scallop lang, as manually defining logical
  constraints would be a pain here.
- Will create formalised logic definitions of non-fuzzy relations manually
  embedded in Expt1:
  - Numbers with Lines:
    ```math
    \forall x ( Numeral(x) \to \exist HasLine(x) \oplus \exist NoLine(x) )
    ```

  - Numbers with Loops:
    ```math
    \forall x ( Numeral(x) \to \exist HasLoop(x) \oplus \exist NoLoop(x) )
    ```

  - Where:
    ```math
    Zero(x) \to NoLine(x) \land HasLoop(x) \\
    One(x) \to HasLine(x) \land NoLoop(x) \\
    Two(x) \to NoLine(x) \land NoLoop(x) \\
    Three(x) \to NoLine(x) \land NoLoop(x) \\
    Four(x) \to HasLine(x) \land NoLoop(x) \\
    Five(x) \to HasLine(x) \land NoLoop(x) \\
    Six(x) \to NoLine(x) \land HasLoop(x) \\
    Seven(x) \to HasLine(x) \land NoLoop(x) \\
    Eight(x) \to NoLine(x) \land HasLoop(x) \\
    Nine(x) \to HasLine(x) \land HasLoop(x)
    ```

### Implementation Details

- The logic has been successfully implemented using the `scallopy` Python
  library to embed differentiable logic within the PyTorch training loop.
- Two distinct semantic loss variants were created:
  - **Boolean Semantic Loss:** Computes MSE over pure True/False boolean
    probability distributions of whether a number contains a line or a loop.
  - **Kleene Semantic Loss:** Explores 3-valued logic (True, False, Undecidable)
    to penalize the model for confusing epistemic uncertainty with ontological
    ambiguity, computing MSE over 3D probability distributions.

### Results

- Training loss ![Training loss plot for experiment
  3](./results/Experiment3/train_loss.png)
- Validation loss ![Validation loss plot for experiment
  3](./results/Experiment3/val_loss.png)
- Test loss ![Test loss plot for experiment
  3](./results/Experiment3/test_loss.png)

### Conclusions

- The experiment was a massive success!
- Embedding logical constraints using Scallop guided the model highly
  effectively, and both Boolean and Kleene semantics provided differentiable
  gradients that dramatically improved the learning outcome on QMNIST.

## Experiment 4 - Continuous Differentiable Logic Relaxations (LTN & Kleene/Belnap Semantics)

### Planning

- This experiment serves as an architectural extension of Experiment 3's logic
  on the QMNIST dataset.
- The goal is to avoid external symbolic execution engines (like Scallop/PySDD)
  and natively evaluate the probabilities and truth values using lightweight
  continuous mathematics in pure PyTorch.
- We implement several continuous logic relaxations, progressing from basic
  Boolean logic to expressive Bilattice (Fuzzy Belnap) logic that natively
  models epistemic uncertainty without non-differentiable logic solvers.

### Implementation Details

Five distinct loss function paradigms were implemented natively in PyTorch: 1.
**Exact Boolean Semantic Loss:** Maps strict true/false constraints into
`(10, 2)` label tensors and natively evaluates boolean constraint probability
across the predicted class distribution. 2. **Logic Tensor Networks (LTN):**
Utilizes fuzzy logical operators (e.g., Lukasiewicz equivalence) to evaluate
continuous tensor satisfaction bounds in `[0, 1]`. 3. **Exact Kleene Semantic
Loss:** Maps the three states of Kleene logic (True, False, Undecidable)
directly into a `(10, 3)` evidence distribution, computing the continuous
Euclidean expectation natively in PyTorch. 4. **Gödel LTN Semantics:** Maps
Kleene's `Undecidable` state to the `0.5` truth value, using Gödel t-norms
(`min`/`max`) to evaluate logic constraints natively on epistemic uncertainty
boundaries. 5. **Fuzzy Belnap Logic (Bilattice LTN):** Evaluates evidence for
*Truth* `(t)` and *Falsity* `(f)` completely independently over two orthogonal
dimensions. This avoids arbitrarily forcing total probability constraints and
handles true logical contradiction (Overdetermined / `(1.0, 1.0)`) and ignorance
(Underdetermined / `(0.0, 0.0)`).

### Results

- The test results natively match and seamlessly track the accuracy metrics
  achieved by the heavy Scallop engine in Experiment 3.
- All models achieved approximately `~81.2% - 83.3%` test accuracy, properly
  internalizing the structural constraints of loops and lines within QMNIST.
- Fuzzy Belnap Logic, in particular, offers extremely smooth continuous
  gradients for training due to the independent tracking of dual-axis evidence.

### Conclusions

- The experiment was a resounding success, proving that complex non-classical
  logic states (such as Kleene's Undecidability or Belnap's Overdetermination)
  can be modeled highly effectively as pure PyTorch continuous tensor operations
  without offloading to discrete symbolic logic engines.

## Experiment 5 - Semantic loss by using SmoothMax to bake in relational embeddings

### Planning

- This experiment serves as an architectural extension of Experiment 1, using
  SmoothMax to bake the relations directly into the loss function
- The goal is to achieve a similar effect to Experiment 1, but to create a
  continuously combined differentiable landscape that can direct training more
  efficiently
- We use pytorch's builtin `logaddexp` function to bake the top hat functions
  from the relational matrix directly into the loss function

### Results

- Test results show these semantic loss functions to provide only negligible
  improvement over naive categorical cross entropy.

### Conclusion

- This experiment appears to be a failure. I don't know why though. My suspicion
  this is probably moreso a matter of optimising `alpha` and `temperature` than
  it is a matter of this not being able to work. But if so, the approach in
  Experiment 1 is likely superior in most cases.
