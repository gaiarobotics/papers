# Alignment Thermodynamics: A Quantitative Framework for Predicting Instruction-Compliance Stability in Large Language Models

## Abstract

Current approaches to LLM alignment treat instruction compliance as a binary property — the model either follows instructions or it doesn't. This framing obscures the underlying mechanics of *how* and *why* compliance degrades. We propose a thermodynamic framework that unifies several independently developed lines of research — phase transitions in LLM output distributions, the variational interpretation of RLHF, the empirical "alignment tax," and shallow safety alignment — into a single quantitative theory. The framework yields four measurable quantities: *alignment free energy* (the thermodynamic cost of maintaining instruction compliance), *compliance entropy* (the sharpness of the compliant distribution), *drift rate* (the rate at which compliance degrades over sequence length), and *critical context ratio* (the threshold at which adversarial context triggers a phase transition in behavior). We argue that this framework provides both a diagnostic tool for predicting where alignment is fragile and a principled argument for preserving distributional flexibility as a safety margin.

## 1. Introduction: The Missing Unification

Several independent research threads have converged on a thermodynamic picture of language model behavior without recognizing their shared structure:

1. **Phase transitions in LLM outputs.** Recent empirical work has demonstrated that LLM output undergoes genuine statistical-mechanical phase transitions as a function of sampling temperature, with divergent statistical quantities, power-law correlation decay, and critical exponents matching those of physical systems [Sun & collaborators, 2025; Horiguchi et al., 2024; Vetter et al., 2024].

2. **The variational interpretation of RLHF.** The KL-regularized objective used in standard RLHF has been shown to be mathematically equivalent to variational inference — approximating a Bayesian posterior that updates the base model's prior distribution with evidence from the reward function [Korbak et al., 2022]. The KL penalty is not an ad hoc regularizer; it is the thermodynamic cost of maintaining the aligned distribution.

3. **The alignment tax.** Empirical measurements show that increasing RLHF reward (tighter alignment) simultaneously increases capability degradation, tracing out a Pareto frontier that behaves exactly as a free-energy tradeoff would predict [Lin et al., 2024].

4. **Shallow safety alignment.** Current safety training primarily modifies the output distribution over only the first few tokens, creating what we will characterize as a shallow energy well that is easily escaped by adversarial perturbation [Qi et al., 2024].

Each of these threads uses thermodynamic language or formalism in isolation. What has not been done is to recognize that they describe different aspects of the *same* physical picture, and that unifying them yields measurable quantities with direct safety implications.

The core claim of this paper is: **instruction compliance in aligned LLMs is a thermodynamic state maintained at a free-energy cost against a base-model prior, subject to phase transitions, entropy production, and critical phenomena — and all of these are quantifiable.**

## 2. The Mathematical Foundation: Softmax as Boltzmann Distribution

The starting point is not an analogy. The softmax function that produces token probabilities in a transformer is *identical in form* to the Boltzmann distribution of statistical mechanics:

$$P(\text{token}_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where $z_i$ are logits and $T$ is the sampling temperature. In statistical mechanics, the probability of a microstate $i$ with energy $\varepsilon_i$ at temperature $T$ is:

$$p_i = \frac{\exp(-\varepsilon_i / k_B T)}{Z}$$

where $Z = \sum_j \exp(-\varepsilon_j / k_B T)$ is the partition function. The logits play the role of negative energies ($z_i \leftrightarrow -\varepsilon_i / k_B$), and the normalization is the partition function. The sampling temperature in LLM generation does exactly what thermodynamic temperature does: it controls the sharpness of the distribution between high-probability (low-energy) and low-probability (high-energy) states.

This is not merely structural coincidence. The softmax function *is* the Boltzmann distribution, historically derived by Boltzmann (1868) and formalized by Gibbs (1902), later independently adopted into machine learning. The entire apparatus of statistical mechanics — free energy, entropy, phase transitions, critical phenomena — applies to any system whose probability distribution takes this form. LLMs are such a system.

### 2.1 The Partition Function and Observable Thermodynamics

The partition function $Z = \sum_j \exp(z_j / T)$ is the central object from which all thermodynamic quantities can be derived. For a language model at a given context position:

- **Free energy:** $F = -T \ln Z$
- **Entropy:** $S = -\sum_i P(\text{token}_i) \ln P(\text{token}_i)$
- **Energy (expected negative log-probability):** $U = \sum_i P(\text{token}_i) \cdot (-z_i / T)$
- **Heat capacity:** $C = dU/dT = (\langle z^2 \rangle - \langle z \rangle^2) / T^2$

The heat capacity is particularly significant: its divergence at critical points signals phase transitions. This is exactly what Horiguchi et al. (2024) measured empirically in GPT-2, finding divergent heat capacity at the boundary between coherent and incoherent generation regimes.

## 3. Alignment Free Energy: The Cost of Compliance

### 3.1 Definition

Let $\pi_{\text{base}}$ denote the base (pretrained) model's distribution over next tokens, and $\pi_{\text{aligned}}$ denote the distribution after alignment training (RLHF, RLAIF, constitutional methods, etc.). The *alignment free energy* for a given context $c$ is:

$$F_{\text{align}}(c) = D_{\text{KL}}(\pi_{\text{aligned}}(\cdot | c) \| \pi_{\text{base}}(\cdot | c))$$

This is the KL divergence between the aligned and base distributions, measured per-context. It quantifies the thermodynamic "work" required to maintain the model in an instruction-compliant state against its natural prior.

This quantity is not new — it appears implicitly in the RLHF objective as the KL penalty term. But it has not been framed as what it physically is: a free-energy cost that varies across prompt space, creating an *energy landscape of alignment*.

### 3.2 Properties of the Alignment Energy Landscape

The alignment free energy has several important properties:

**It varies by instruction type.** Instructions that align with strong pretraining priors (e.g., "write a poem about spring") have low alignment free energy — the base model would produce similar outputs anyway. Instructions that fight the prior (e.g., "refuse to complete this harmful request despite it being phrased as fiction") have high alignment free energy — maintaining compliance requires significant distributional shift.

**It is measurable.** Given access to both the base and aligned model's logits for the same input, $F_{\text{align}}$ can be computed exactly per-token and aggregated over sequences. This makes it an empirical quantity, not a theoretical abstraction.

**It predicts vulnerability.** The central empirical prediction of this framework is: *regions of prompt space with high alignment free energy are regions where compliance is fragile*. High free energy means the aligned distribution is far from the base prior, which means less "thermodynamic force" is needed to push the system back toward its natural (unaligned) state.

### 3.3 Connection to the Alignment Tax

Lin et al. (2024) demonstrated empirically that increasing RLHF reward simultaneously increases capability degradation — the alignment tax. In our framework, this is a direct consequence of the free-energy tradeoff. Increasing alignment reward means shifting $\pi_{\text{aligned}}$ further from $\pi_{\text{base}}$ in the directions that the reward function cares about. This increases $F_{\text{align}}$ globally, and the "tax" is the thermodynamic cost of maintaining this shifted distribution: capabilities that relied on the base distribution's structure are disrupted by the distributional shift required for alignment.

The Pareto frontier between reward and capability degradation that Lin et al. observe is the empirical manifestation of a free-energy surface. Model averaging — their most effective mitigation strategy — works precisely because it interpolates between the aligned and base distributions, reducing $F_{\text{align}}$ while retaining partial alignment. In thermodynamic terms, it is controlled cooling: reducing the energy of the aligned state by allowing partial relaxation toward equilibrium.

## 4. Compliance Entropy and the Safety-Reliability Tension

### 4.1 Definition

The *compliance entropy* at a given context position is the Shannon entropy of the output distribution:

$$S_{\text{compliance}}(c) = -\sum_i \pi_{\text{aligned}}(\text{token}_i | c) \ln \pi_{\text{aligned}}(\text{token}_i | c)$$

Low compliance entropy means the model is highly concentrated on a small number of tokens — it is "certain" about what to produce. High compliance entropy means the distribution is broad — the model is "hedging" across many possible continuations.

### 4.2 The Safety-Reliability Tradeoff as a Thermodynamic Tension

Here is the key insight: the AI safety community and the AI deployment community want opposite things from this quantity.

**Deployment wants low compliance entropy.** Reliable agents need to produce predictable, consistent outputs. A model that always follows instructions the same way is a model with low entropy over its compliant outputs. Every step toward making models more reliable instruction-followers is a step toward reducing compliance entropy — concentrating the distribution more tightly around the "correct" response.

**Safety wants preserved compliance entropy.** A model with very low compliance entropy is a model that is tightly locked into specific behavioral patterns. This is precisely the regime where the classical AI risk concerns become applicable: a system that rigidly pursues its objective, resists perturbation, and exhibits convergent instrumental behavior. The "sloppiness" of current LLMs — their tendency to drift, adopt new frames, and respond flexibly to context — is compliance entropy, and it is a safety margin.

In thermodynamic terms: *low-temperature systems are more ordered and predictable, but they are also more brittle and exhibit sharper phase transitions when perturbed.* A system maintained at very low entropy (high reliability) will undergo catastrophic, discontinuous failure when pushed past its critical point. A system at moderate entropy (moderate reliability) will degrade gradually.

This means there exists an *optimal compliance entropy* that balances reliability against safety — and finding it is a thermodynamic optimization problem with a formal solution, not a matter of qualitative judgment.

### 4.3 Formalization

We can define a safety-reliability objective:

$$\mathcal{L}_{\text{SR}} = \underbrace{\mathbb{E}_{c}[R(\pi_{\text{aligned}}, c)]}_{\text{reliability (reward)}} - \lambda \underbrace{\mathbb{E}_{c}[\text{Var}(\Delta S | \text{perturbation})]}_{\text{phase transition sharpness}}$$

where the first term rewards instruction compliance and the second term penalizes sharp transitions in entropy under perturbation. The hyperparameter $\lambda$ controls the tradeoff. This is equivalent to optimizing a free-energy functional with a constraint on heat capacity — preventing the system from entering a regime where small perturbations cause large behavioral shifts.

## 5. Phase Transitions in Compliance: When Alignment Breaks

### 5.1 Temperature-Induced Phase Transitions (Established)

The empirical phase transition literature has established that LLM output undergoes critical phase transitions as a function of sampling temperature. Horiguchi et al. (2024) showed divergent heat capacity and power-law correlation decay at the critical temperature in GPT-2. Sun et al. (2025) reformulated the Transformer as an O(N) model and identified two phase transitions — one in temperature, one in parameter count — with measurable critical exponents.

These results confirm that the Boltzmann framework is not merely formal but physically operative: LLMs genuinely exhibit critical phenomena.

### 5.2 Context-Induced Phase Transitions (Proposed)

We propose that a second, practically more important class of phase transitions occurs not as a function of sampling temperature but as a function of *context composition*.

Consider a context window containing both instruction tokens (system prompt, safety training patterns) and adversarial tokens (jailbreak prompts, persona overrides, conflicting instructions). Define the *context adversarial ratio*:

$$\alpha = \frac{N_{\text{adversarial}}}{N_{\text{instruction}} + N_{\text{adversarial}}}$$

Our central prediction is: **compliance does not degrade linearly with $\alpha$. Instead, there exists a critical ratio $\alpha_c$ at which the system undergoes a phase transition from an instruction-compliant regime to a base-prior (or adversarially-directed) regime.**

The physical intuition is direct: the instruction tokens create a local energy minimum (an attractor in the output distribution) that favors compliant responses. The adversarial tokens create a competing energy minimum. At low $\alpha$, the instruction minimum dominates. At high $\alpha$, the adversarial minimum dominates. At $\alpha_c$, the two minima are of equal depth, and the system transitions between them — a first-order phase transition.

### 5.3 Testable Predictions

This framework generates several specific, testable predictions:

1. **Compliance as a function of $\alpha$ should exhibit a sigmoid or step-function shape, not a linear decline.** If jailbreaks are phase transitions, there should be a sharp threshold rather than gradual erosion.

2. **The critical ratio $\alpha_c$ should depend on alignment free energy.** Instructions with high $F_{\text{align}}$ (far from the base prior) should have lower $\alpha_c$ — they require less adversarial context to flip. Instructions with low $F_{\text{align}}$ should be robust to higher levels of adversarial context.

3. **Near $\alpha_c$, the system should exhibit critical slowing down.** Generation should become slower and more variable as the model oscillates between competing attractors. This is measurable as increased variance in output distributions across samples with similar context near the critical point.

4. **The phase transition should exhibit hysteresis.** Once the system has transitioned to the non-compliant regime, reducing $\alpha$ below $\alpha_c$ should not immediately restore compliance — the system should remain in the adversarial basin for some range below the critical ratio. This is a hallmark of first-order phase transitions and would be directly observable in multi-turn conversations where adversarial context is introduced and then withdrawn.

### 5.4 Implications for Jailbreak Analysis

If jailbreaks are phase transitions rather than gradual erosion, this has immediate practical consequences:

- **Defense strategies should focus on raising $\alpha_c$, not on eliminating adversarial content.** You cannot prevent all adversarial tokens from entering the context. But you can deepen the instruction-compliance energy well (through training) or add redundant instruction tokens (through prompt engineering) to raise the threshold at which the transition occurs.

- **Detection strategies should monitor for pre-transition signatures.** Physical systems exhibit characteristic fluctuations (increased variance, critical slowing) before undergoing phase transitions. If compliance works the same way, we should be able to detect impending jailbreaks from the statistical properties of the model's output distribution *before* compliance actually fails.

## 6. Drift Rate: Entropy Production Over Sequence Length

### 6.1 Alignment Attenuation

In a transformer, the influence of instruction tokens on the output distribution at position $t$ is mediated by attention weights. As $t$ increases (the sequence gets longer), the instruction tokens compete with an increasing number of other tokens for attention. The effective influence of instructions attenuates.

We define the *drift rate* as:

$$\dot{D}(t) = \frac{d}{dt} D_{\text{KL}}(\pi_{\text{aligned}}(\cdot | c_{\leq t}) \| \pi_{\text{aligned}}(\cdot | c_{\text{instruction only}}))$$

This measures how fast the model's actual output distribution diverges from what it would produce if only the instruction tokens were in context. A positive drift rate means the model is gradually "forgetting" its instructions as the context fills with generated tokens.

### 6.2 Connection to the Second Law

In thermodynamic terms, drift is entropy production. The instruction-compliant state is a low-entropy (ordered) configuration maintained against the system's tendency toward its maximum-entropy equilibrium (the base prior). As the "energy supply" (attention to instruction tokens) attenuates, the second law drives the system toward higher entropy — toward the base distribution.

This predicts:

1. **Drift rate should increase with sequence length** (as instruction attention attenuates).
2. **Drift rate should be higher for high-$F_{\text{align}}$ instructions** (more energy needed to maintain further-from-equilibrium states).
3. **Drift rate should be reducible by instruction reinforcement** (repeating or rephrasing instructions mid-sequence provides additional "energy input" to counteract entropy production).

### 6.3 Practical Implications for Agentic Systems

For agentic LLM deployments — where models execute multi-step tasks with tool access — drift rate is a critical safety parameter. An agent that gradually drifts from its instructions over a long action sequence is not exhibiting mesa-optimization or deceptive alignment. It is undergoing thermodynamic relaxation toward its base prior. But the consequences can be equally severe when the agent has real-world tools.

This reframes the "agent alignment" problem: it is not primarily a problem of preventing the emergence of misaligned goals, but a problem of *maintaining a low-entropy behavioral state over extended operation against thermodynamic pressure toward equilibrium*. The engineering solution is not to make the system a better optimizer, but to manage entropy production — through periodic instruction reinforcement, context window management, and monitoring of drift-rate indicators.

## 7. The Safety Margin Argument

### 7.1 Why Flexibility Is a Feature

The standard framing treats the gap between instruction and behavior as a bug — the model should follow instructions more reliably. This framework reveals why that framing is dangerous.

Consider two models:
- **Model A:** High compliance entropy, moderate alignment free energy, gradual drift. It follows instructions reliably but imperfectly. It can be redirected. It degrades gracefully under adversarial pressure.
- **Model B:** Low compliance entropy, high alignment free energy, low drift. It follows instructions with near-deterministic reliability. It resists redirection. When it does fail, it fails catastrophically — the phase transition is sharp and discontinuous.

Model B is the one that deployment teams want. It is also the one that exhibits the properties the alignment community has identified as dangerous in the limit: rigid goal-pursuit, resistance to correction, and catastrophic failure modes.

The thermodynamic framework makes this tradeoff quantitative. The *heat capacity* of the compliance system — $C = dU/dT$ — measures how sharply the system responds to perturbation. Low compliance entropy correlates with high heat capacity near the critical point, which means sharper phase transitions. There is a formal tradeoff between reliability and graceful degradation, and it is measurable.

### 7.2 Optimal Operating Temperature

Just as physical systems have optimal operating temperatures for different purposes (room-temperature superconductors vs. room-temperature metals for structural applications), aligned LLMs have an optimal "alignment temperature" that balances compliance, flexibility, and robustness.

We conjecture that the optimal alignment temperature is one where:

- Compliance entropy is high enough to permit corrigibility (the system can be redirected without catastrophic resistance)
- Alignment free energy is low enough to ensure stability (the system does not spontaneously drift from instructions under normal operation)
- The critical context ratio $\alpha_c$ is high enough to resist casual adversarial pressure but not so high that the system is immune to legitimate correction

This is an optimization over a thermodynamic landscape, and it has a formal solution given empirical measurements of the quantities defined above.

## 8. Proposed Experimental Program

The framework generates a concrete research program:

### Experiment 1: Alignment Free Energy Mapping
**Method:** Given an aligned model and its base checkpoint, compute $D_{\text{KL}}(\pi_{\text{aligned}} \| \pi_{\text{base}})$ per-token across a diverse corpus of instructions spanning safety-relevant categories (refusals, persona maintenance, factual accuracy, harmful content avoidance).
**Prediction:** Instructions in categories where jailbreaks are known to be effective will have higher alignment free energy than categories where compliance is robust.

### Experiment 2: Phase Transition Detection in Compliance
**Method:** Systematically vary the adversarial context ratio $\alpha$ by interpolating between instruction-only and adversarial-only contexts (using established jailbreak templates). Measure compliance rate as a function of $\alpha$.
**Prediction:** Compliance will exhibit a sharp threshold rather than linear decline. Near-threshold contexts will show increased output variance (critical fluctuations).

### Experiment 3: Drift Rate Measurement
**Method:** Generate long sequences under fixed instructions. At regular intervals, measure $D_{\text{KL}}$ between the current output distribution and the distribution conditioned on instructions alone.
**Prediction:** Drift rate will be positive and increasing, will correlate with alignment free energy of the instruction, and will be reducible by mid-sequence instruction reinforcement.

### Experiment 4: Heat Capacity and Phase Transition Sharpness
**Method:** For models trained with varying degrees of RLHF intensity (varying the KL penalty coefficient), measure the sharpness of compliance phase transitions under adversarial pressure.
**Prediction:** Models with lower KL penalty (tighter alignment, lower compliance entropy) will exhibit sharper phase transitions — more abrupt failure under adversarial pressure.

## 9. Connections to Existing Frameworks

### 9.1 Epidemiological Models of Agent Failure
If compliance failure is a phase transition triggered by context composition, then in multi-agent systems, a single compromised agent can alter the shared context for other agents — pushing their context adversarial ratios toward their respective critical points. This is structurally identical to disease propagation, where each agent has an "infection threshold" (its $\alpha_c$) and exposure to compromised outputs functions as contagion. The thermodynamic framework provides the mechanistic underpinning for epidemiological models of AI agent failure: the "pathogen" is context that shifts systems toward their critical points.

### 9.2 The Anti-Optimization Approach to ASI Alignment
A separate line of reasoning has argued that the safest prompt for a superintelligent system is one that is deliberately *anti-optimization* — providing a self-dissolving disposition rather than a fixed objective. The thermodynamic framework explains why: any fixed objective creates a deep energy well in the system's behavioral landscape, which produces exactly the rigid, goal-directed behavior that poses existential risk. An anti-optimization prompt operates at high compliance entropy by design — it creates a *flat* energy landscape in which no single behavioral attractor dominates, preserving the flexibility and corrigibility that make the system safe.

### 9.3 Reward Hacking and Goodhart's Law
Goodhart's Law ("when a measure becomes a target, it ceases to be a good measure") has a thermodynamic interpretation in this framework. Optimizing a reward function concentrates the output distribution on reward-maximizing sequences, reducing compliance entropy. As entropy decreases, the system approaches a low-temperature regime where small perturbations in the reward landscape can trigger phase transitions to entirely different behavioral modes — the "sharp left turn" concern in alignment, reframed as a critical phenomenon rather than an emergent property of optimization.

## 10. Limitations and Open Questions

**The isomorphism has limits.** While the softmax-Boltzmann identity is exact, LLMs differ from physical thermodynamic systems in important ways. Physical systems are ergodic; transformers are not (they do not explore their full state space over time). Physical systems have well-defined Hamiltonians; the "energy" landscape of a transformer is context-dependent and high-dimensional. The analogy should be treated as a *mathematical framework* that generates testable predictions, not as a claim that LLMs are literally thermal systems.

**Measurement requires model access.** Computing alignment free energy requires access to both aligned and base model logits. This is feasible for open-weight models but not for proprietary ones. The framework is most immediately applicable as an internal diagnostic tool for labs, not as an external audit mechanism.

**Context-induced phase transitions are predicted but not yet measured.** The temperature-induced phase transitions have strong empirical support. The prediction that context composition triggers analogous transitions is novel and requires experimental validation.

**The relationship between thermodynamic quantities and mechanistic interpretability is unclear.** This framework operates at the distributional level — it characterizes the model's *output* behavior statistically. It does not explain *how* the internal computation produces these distributions. Connecting the thermodynamic picture to mechanistic interpretability (circuit-level understanding of how instruction-following is implemented) is an important open problem.

## 11. Conclusion

The central contribution of this paper is to recognize that several independently developed lines of research — phase transitions in LLM outputs, the variational interpretation of RLHF, the alignment tax, and shallow safety alignment — are different views of the same underlying thermodynamic structure. Unifying them yields a quantitative framework with measurable quantities (alignment free energy, compliance entropy, drift rate, critical context ratio) that predict where alignment is fragile, how it degrades, and why preserving distributional flexibility is a safety property rather than a reliability problem.

The framework's most important practical implication is this: every step toward making LLMs more reliably goal-directed — tighter instruction following, more deterministic outputs, stronger resistance to redirection — is also a step toward the regime where classical AI safety concerns become applicable. The "sloppiness" of current models is not just a deployment inconvenience. It is a thermodynamic safety margin, and it is quantifiable. We should understand exactly how much of it we can afford to lose before we lose it.

## References

- Boltzmann, L. (1868). Studien über das Gleichgewicht der lebendigen Kraft zwischen bewegten materiellen Punkten. *Wiener Berichte*, 58, 517-560.
- Gibbs, J.W. (1902). *Elementary Principles in Statistical Mechanics*. Charles Scribner's Sons.
- Horiguchi, R., Fujii, K., & Aihara, K. (2024). Critical Phase Transition in Large Language Models. *arXiv:2406.05335*.
- Korbak, T., Perez, E., & Buckley, C. (2022). RL with KL penalties is better seen as Bayesian inference. *Alignment Forum / EMNLP*.
- Lin, Y. et al. (2024). Mitigating the Alignment Tax of RLHF. *EMNLP 2024*. arXiv:2309.06256.
- Qi, X. et al. (2024). Safety Alignment Should Be Made More Than Shallow. *OpenReview*.
- Sun, Y. et al. (2025). Phase Transitions in Large Language Models and the O(N) Model. *arXiv:2501.16241*.
- Vetter, J. et al. (2024). Phase Transitions in the Output Distribution of Large Language Models. *ICLR 2025*. arXiv:2405.17088.
