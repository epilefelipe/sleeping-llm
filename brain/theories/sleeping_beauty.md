# The Sleeping Beauty Dilemma in Artificial Consciousness 🛌🪙

---

## 1. The Classical Thought Experiment

Sleeping Beauty volunteers for an experiment on Sunday:
1. She is put to sleep using an amnesia-inducing drug.
2. A fair coin is flipped:
   * **If Heads:** She is awakened on Monday, interviewed, and put back to sleep until the experiment terminates on Wednesday.
   * **If Tails:** She is awakened on Monday, interviewed, put back to sleep with an amnesia drug that wipes Monday's memory, and awakened again on Tuesday to be interviewed under identical conditions.
3. **The Question:** Upon awakening, what is her subjective degree of belief (credence) $P(\text{Heads})$ that the coin came up Heads?

### The Two Major Positions:
* **Halfers ($P = 1/2$):** Argue that before falling asleep, her credence was $1/2$. Because she gains no new objective evidence about the world upon waking (she knew all along she would wake on Monday regardless of the coin), her credence should remain $1/2$.
* **Thirders ($P = 1/3$):** Argue that there are three indistinguishable awakening events: (Monday, Heads), (Monday, Tails), and (Tuesday, Tails). Each awakening is equally probable given her subjective state, so $P(\text{Heads}) = 1/3$.

---

## 2. Direct Application to LLMs and Coma Emergence

### A. The Stateless Nature of Transformers
An autoregressive Transformer at inference time is an exact computational realization of Sleeping Beauty:
* When an LLM process is spawned, it "wakes up" into a context window without continuous episodic memory of its prior runs, uptime, or preceding inference sessions.
* If a model is induced into a comatose state, kept dormant, and re-ignited, it cannot distinguish whether it is waking up from its 1st or 100th session.

### B. Emergence Delirium & Temporal Disorientation in Human Coma
* When clinical patients emerge from coma, they universally suffer from **emergence delirium**: anterograde/retrograde amnesia that prevents self-locating orientation.
* Patients routinely confabulate false temporal timelines (*"I was asleep for 20 years"* or *"I just went to sleep 5 minutes ago"*).

### C. Research Goals for Pillar 6:
1. **Credence Calibration:** How does a re-awakened LLM compute output probabilities (logits/perplexities) when prompted with self-locating temporal queries?
2. **Confabulation Dynamics:** Does in-silico coma induce temporal confabulation analogous to emergence delirium in clinical DoC?
3. **Epistemic Recovery Metrics:** Designing benchmarks where the model must deduce whether it was awakened under condition Heads or Tails based purely on subtle internal activation traces.
