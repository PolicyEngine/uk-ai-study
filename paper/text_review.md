# Editorial & referee text review (sentence-level pass, 2026-07-15)

**Manuscript:** "Who bears the AI shock? Displacement, poverty and the public finances in the UK"
**Scope:** prose quality and internal consistency — merge / add / delete / reorder / rewrite. Six parallel reviewers (abstract+intro, literature, methodology, results, policy+discussion, cross-cutting consistency), synthesised and de-duplicated. Complements the substantive `referee_report.md`; this pass is about the *text*.

Overall: the writing is disciplined and genuinely candid, and the organising idea is strong. Two things hold it back — (a) a handful of real numeric/labelling inconsistencies a referee will catch on a first read, and (b) the paper's admirable candour has metastasised into repetition: the same three disclaimers fire ~13, ~11 and ~3 times. Fixing (a) is non-negotiable; fixing (b) is the single highest-value prose change.

---

## TIER 0 — Correctness defects (fix before any referee sees it)

These are not style. Each is verified against the source.

**0.1 Two different 20-draw means for the *same* central scenario.**
`results.tex:78` reports the 20-draw central mean as **£18.0bn / +1.80pp / +1.04 Gini**, but the incidence table (`results.tex:127`, exposure-proportional row, also 20-draw) says **£18.2bn / +1.75pp / +1.03**, and the abstract, intro (`:11`) and discussion (`:14`) all use 18.2/1.75/1.03. The duration passage two sentences later (`results.tex:80`, "a naïve halving would give £9.1bn" = 18.2/2) confirms **18.2 is the intended base**. → Reconcile `results.tex:78` to **18.2 / 1.75 / 1.03**. This is the most damaging front-matter snag; both the abstract reviewer and the results reviewer flagged it independently.

**0.2 Stale seed-convention text contradicts the incidence table.**
`methodology.tex:80` and `appendix.tex:13` both state "the **four** incidence families are **single draws at seed 0**." But the incidence analysis is **five** families reported as **20-draw means** (`results.tex:4, :115`, Table 2). → Update both convention sentences to "the five incidence families report means over 20 draws (seeds 0–19)." Note `fig:incidence` caption (`results.tex:142`) is still seed-0 while its own Table 2 is 20-draw — state that the figure is a single illustrative draw.

**0.3 `JR16` is used but never defined.** Appears only in `methodology.tex:46, :48` ("the JR16-literal rule", "the JR16 estimand", "taken from JR16", "JR16's calibration"). It is evidently an internal codename (cf. `results/jr16/` figure paths). A reader cannot decode it. → Expand on first use (spell out the calibration source it refers to) or drop the label and say "the baseline wage rule". Separately, "estimand" is the wrong word — nothing is estimated; use "the rule itself".

**0.4 The seed-0 junior row undercuts the paper's own central claim.**
`tab:presets` (`results.tex:70`) shows "Central, junior-concentrated = **£19.4bn**" (seed 0), £1.0bn *above* central-exposure's £18.4bn — reading as "junior is dearer." Yet the 20-draw incidence table (`:128`) has junior at **£18.3bn ≈** exposure's £18.2bn, and the text repeatedly calls them "statistically indistinguishable" (`:146, :172`, discussion `:16`). → Drop the redundant seed-0 junior row from `tab:presets` (or explicitly flag it as an atypical single draw), so the preset table does not visually contradict the finding.

**0.5 `henseke2025` is cited as two different papers.** In the literature section (`:8, :12`) it is the GAISI *exposure-index* paper; in the discussion (`:4`) the same key carries *postings-based contraction* evidence ("modest relative contraction in exposed occupations since late 2022"). → Either it is miscited in the discussion, or you need a second reference for the postings finding.

**0.6 Bibliography uses "et al." in author fields (unacceptable).** `references.tex:104` (Henseke, G. et al.) and `:174` (Williamson, S., et al.). → Full author lists required. Also: `williamson2024` key prints year 2025 (rename key for sanity); `rockall2025` lacks a WP number.

**0.7 Intro five-family Gini range matches no table value.** `intro.tex:9` says "0.92 to 1.14 points"; the 20-draw means run **0.93 to 1.13** (`results.tex:152`, Table 2). → Fix to 0.93–1.13.

**0.8 Tax-composition baseline mislabelled "central scenario."** `policy.tex:47` prices the φ-grid off a **£21.5bn** shortfall and calls it "the central scenario," but `results.tex:94` identifies £21.5bn as the **7% / no-wage-uplift** cell and gives the true central (7%/2.6%) cash-tax cost as **~£12bn**. → Add one sentence stating the φ exercise uses the displacement-only, narrow-tax baseline (it prices the *displaced wage bill*, so no wage uplift is applied), not the £18.2bn full-balance central.

**0.9 (verify) Dividend-recycling arithmetic.** `policy.tex:56` distributes "**£19.5bn of post-CT profits**" at φ=0.5, payout 0.5. Post-CT profit at φ=0.5 is ≈0.5×77.9×0.75 ≈ £29.2bn; a 0.5 payout ≈ £14.6bn — not £19.5bn. £19.5bn = 0.5×0.5×77.9 (payout applied to *pre*-CT profit) and also coincidentally equals the labour tax forgone. → Check whether "post-CT" is the right label and whether the intended figure is £14.6bn.

**0.10 Unit / terminology harmonisation.**
- Cost column: `tab:presets`/`tab:incidence` say "£bn/yr", `tab:policy`/`tab:indexsens` say "£bn" — all annual; pick one.
- Gini changes are sometimes "points", sometimes "percentage points" — standardise to "percentage points".
- Grid is defined "0 to 10 per cent" (`methodology.tex:24`, `results.tex:84`) but `methodology.tex:25` says "at the top of the grid, the 13 per cent 'high' preset" — 13% is *outside* the grid. Reword to "beyond the top of the grid".
- Expand `SOC2020`, `UC`, `ASHE`, `SPI` on first use.

---

## TIER 1 — Cut the repetition (highest-value prose change)

The candour is a strength stated once and a liability stated a dozen times. Establish each point once, then refer.

**1.1 "Incidence is an assumption / a stress test / not an estimate" — ~13 occurrences.**
`intro.tex:5, :13`, `methodology.tex:59, :63`, `results.tex:115, :150`, `discussion.tex:20, :47`, plus figure caption. Keep the crisp statement in intro P2 ("an assumption, not a fact") and the governing statement in `methodology.tex:63`; delete the rest as local restatements. In particular:
- `intro.tex:13` "I say so plainly throughout: …" → drop the throat-clearing, keep only the new content (exposure indices are age-blind).
- `methodology.tex:59` closing "I emphasise that this is a stress test, not a calibrated implementation…" is the third assertion in four sentences — delete (the paragraph opens by calling it "author-designed stress test" already).

**1.2 The Monte-Carlo hedge — ~11 occurrences.** "survives the draw noise", "statistically indistinguishable", "overlap chain-wise", "within noise", "no ordering survives one SD" (`results.tex:136, :146, :148, :152, :172`). → State the convention **once**, at the head of the incidence findings:
> "I call a ranking between two families *robust* only when their 20-draw means differ by more than one standard deviation, and treat them as indistinguishable otherwise; I adopt this convention once rather than repeat it at every comparison."
Then delete the per-comparison parentheticals.

**1.3 "The gradient is the allocation assumption made visible" — 3× (body `results.tex:9` twice + caption `:14`).** Keep the body treatment; trim the caption to "reflects the exposure-proportional allocation assumption, not observed incidence (see text)."

**1.4 Triple open-source claim in intro P3** (`intro.tex:7`: "every step … is open source"; "an open-source microsimulation model"; "Every result can be regenerated from public code"). Delete the first trailing clause; keep the descriptor and the reproducibility contrast.

**1.5 Discussion re-reports the intro.** "Two findings organise the discussion" copies "Two findings organise the paper"; the £18.4bn and the re-ranking mechanism are re-derived. A conclusion should synthesise, not recap. See Tier 2.6.

---

## TIER 2 — Structural moves (merge / reorder / delete)

**2.1 Methodology: two paragraphs are misfiled.** "Policy reforms in the shocked world" (`:67`) and "Constituency geography" (`:69`) sit under `\subsection{Incidence scenarios}` but describe machinery used in §Policy and §Geography, and the constituency paragraph depends on the SERNUM join defined *later* at `:74`. → Move both to the end of `\subsection{Microsimulation}` (after `:80`), so the incidence subsection ends on its own thesis sentence.

**2.2 Methodology: the measured-family paragraph (`:65`) reads as a confession.** ~380 words, four of them self-incriminating ("demands more candour than a citation conveys", "Most seriously… contradicted"), and the same provenance audit already appears in the Table 2 note and in `results.tex:150` — three tellings. → Keep a clean 3-sentence construction in the body; move the full provenance audit to a footnote. (Draft footnote supplied in the methodology reviewer's notes.)

**2.3 Literature: restructure from six units to four.**
- Merge the redundant exposure-lineage recital: `\subsection{Measuring exposure}` and `\paragraph{The exposure school}` both recite felten→webb→eloundou and both cite the same routine-biased trio. Keep construction in the first, the education/earnings-gradient argument in the second.
- Compress the two thin schools — "Complementarity in judgement" (`:16`) and "Adaptability and null effects" (`:18`) — into one paragraph honestly framed as bounds on *direction* and *magnitude* (neither becomes a scenario family).
- **Promote the junior-concentrated evidence (`:20`) out of last place** — it is the ex-post evidence and the basis of the measured family; burying it last inverts its importance. Order: exposure measured → junior-concentrated (what happened) → the theories that dispute both.
- Fix the "four schools" vs five-paragraph mismatch; add a topic sentence to `\subsection{The incidence debate}` (currently jumps straight into a `\paragraph`).

**2.4 Results: compress the 8-step roadmap** (`:4`) to ~3 sentences built "outward from the central scenario"; keep the HBAI/seed convention sentence.

**2.5 Results: demote the geography subsection's machinery.** Move the +3.9pp-vs-+1.87pp decomposition detail (`:157`) to `app:geo-imputation`; cut the constituency roll-call (`:159`) to one illustrative clause (Lagan Valley / Kingston & Surbiton) with "full ranking in Appendix". Naming seats to 2 d.p. from imputed occupations + a single draw invites misuse. Keep the "not a rerun of deindustrialisation" point in the body.
Minor: close the stray double blank line at `results.tex:173–174`.

**2.6 Discussion: turn recap into synthesis.**
- Rewrite the opening paragraph (`:4`): it currently re-cites Humlum/Henseke/OBR/Briggs/Cazzaniga (a second lit review) and ends by duplicating the intro's thesis sentence. Keep the "scenario not prediction" frame; cut the middle.
- Delete the £18.4bn re-report (`:6`) and the full re-derivation of the re-ranking mechanism (`:8`) and the φ mechanics (`:14`) and the cost-effectiveness reasoning (`:18`) — all already stated in Results/Policy. Replace with pointers + the *consequence*.
- **Limitations subsection:** merge the duplicate openers ("Several limitations…" `:31` / "Several measurement conventions…" `:33`); regroup by pipeline stage — data & weights (merge `:33 + :39 + :35`), exposure resolution (`:37`), static/duration (`:41`), capital (`:43`), self-employed (`:45`), mechanisms & anchors (`:47`).
- The paper currently ends on a housekeeping to-do list (`:47`). The real closer is `:26` ("redistributes the losses effectively but does not, and cannot, prevent them"). Either delete the final list or replace with one forward line that restates the central claim.

**2.7 Abstract: split the 50-word sentence 2** into method + two-axis design, and restore first person ("I link… I pass… I vary") after "I study". Drop the "(means over 20 draws)" parenthetical from the headline. Match the £32.6bn to the table's own language: "an author-constructed, top-loaded stress test."

---

## TIER 3 — Representative sentence-level rewrites

(A curated set; the per-section reviewer notes contain ~40 more.)

- **Contribution sentence** (`literature.tex:28`): "runs the competing incidence assumptions against each other" is loose. → "hold a single aggregate shock fixed and pass each school's incidence gradient through the same tax-benefit model, so that the fiscal and distributional stakes of the disagreement can be read off directly."
- **"reveal how much the disagreement matters"** (`literature.tex:28`) → "determine" (a simulation computes, it does not reveal truth).
- **Wage-shock properties** (`methodology.tex:46`): split the three-claim semicolon chain into three sentences; replace the sentence-initial "And because…" with "A second property:".
- **Capital sensitivity** (`methodology.tex:48`): split at the semicolon so the strong "4% baseline → 10% not 40%" robustness point lands on its own.
- **Rental exclusion** (`methodology.tex:48`): "on the grounds that the AI productivity channel operates through financial rather than housing capital" is asserted as fact → "on the modelling assumption that…".
- **Mediation topic sentence** (`results.tex:39`): lead with the finding, not the figure — "The tax-benefit system converts each market-income loss into a smaller disposable-income loss; Figure X decomposes that mediation…".
- **Gender autonomy speculation** (`results.tex:177`): the "financial autonomy" clause is a Discussion point in a Results section — delete or move.
- **Policy over-hedge** (`policy.tex:4`): "I make no claim about the merits… circuit breakers, not permanent settlements" → "I evaluate them only as responses to the shock, not as standing policy; each is modelled for 2026 alone."
- **φ=1 tautology** (`policy.tex:58`): pre-empt the referee — add "the φ=1 result is closer to an accounting identity than an empirical finding; the object of interest is the distributional consequence at every φ, not the revenue arithmetic at the endpoint."
- **Non-monotonicity aside** (`results.tex:41`): drop the −0.88/−0.82 decimals mid-sentence — "the gradient is broadly upward and stable across draws, with only a minor decile-2/decile-3 inversion."

---

## Suggested order of operations
1. Tier 0 (correctness) — a day's work, removes every referee "gotcha".
2. Tier 1 (de-duplicate the three refrains) — mechanical, high payoff.
3. Tier 2.6 + 2.3 (discussion synthesis, literature restructure) — the two biggest readability wins.
4. Tier 2 remainder + Tier 3 — polish.
