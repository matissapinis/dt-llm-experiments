# dt-llm-experiments

A framework and accompanying analyses for systematically studying large language model (LLM) behavior in decision-theoretic and anthropic-reasoning problems. The repository accompanies two theses by [Matīss Apinis](https://github.com/matissapinis) at the University of Latvia.

The repository is organised into two self-contained sub-projects, one per thesis:

## [`dt-llm-bsc/`](./dt-llm-bsc)

**Bachelor's thesis** (May 2025): LLM behavior in Newcomb-like decision problems with a focus on the contrast between causal (CDT) and evidential (EDT) decision theory recommendations. Original framework, BSc-era runners, BSc problem configurations and an example experiment notebook. See [`dt-llm-bsc/README.md`](./dt-llm-bsc/README.md) for setup and usage.

## [`anthropics-llm-msc/`](./anthropics-llm-msc)

**Master's thesis** (May 2026): *Epistemic Behavior of Large Language Models in Anthropic Reasoning Problems* — extends the framework to anthropic-reasoning problems (*Sleeping Beauty*, *Incubator*, standard *Doomsday*, and "past" *Doomsday* argument) with *Self-Sampling Assumption* (SSA) and *Self-Indication Assumption* (SIA) as the normative principles. Includes the MSc Stage 1b dataset (13 824 responses across 12 model/reasoning-mode configurations), the two-stage parser, sanity-check rounds, and all analysis scripts that substantiate the quantitative claims in the thesis. See [`anthropics-llm-msc/README.md`](./anthropics-llm-msc/README.md) for setup and usage.

## Notes

Each sub-project is intentionally self-contained — it carries its own `src/`, configuration, data archives, and README so that either can be cloned, set up and replicated independently of the other. The two sub-projects share a common methodological ancestry (descriptive question types about decision-theoretic / anthropic reasoning principles, plus expressed-attitude questions; algorithmically variated problem templates) but use different problem families, normative principles, and model panels.

## License

Code and configurations are made public for research replication and educational reference.
