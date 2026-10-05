# Figure legends

## Figure 1 | Threadfin links BCR families to captured germinal-centre states

**A,** Germinal-centre (GC) organisation. The schematic is original artwork by Chen Satoshi. Arrows depict established GC biology and are not estimates from Threadfin. **B,** A conceptual trace of the same 18 paired cells, allocated to three sequence-defined families (A–C; six cells per family), from scRNA states and scBCR sequences to family profiles and a clone map. The scBCR sequence-space sketch illustrates sequence neighbourhoods used to establish family membership; it is not a learned BCR encoder or an analysis embedding. In the v4 analysis, donor-restricted IGH families are inferred from sequence only, and context-adjusted receptor-excluded RNA kernel profiles separately summarise their captured state distributions. The two displayed spaces are not concatenated UMAPs. Ring charts are teaching representations of state composition; one point represents one family, and proximity represents similarity of captured RNA-state profiles. Positions and fractions are illustrative. **C,** Four visual evidence cards connect clone profiles to the appropriate measurements: reporter mCherry-low (at least six divisions) versus mCherry-high (fewer than six divisions) GC-state bias, same-mouse GC–PB receptor sharing in *Plasmodium*, repeated capture of a human family, and clone coherence relative to shuffled family labels across 12 analyses. The separate illustrative family letters in C identify examples within each study and do not connect clones across mouse and human experiments. These outputs describe captured observations at their respective experimental resolutions; they do not validate a future-fate prediction.

## Figure 2 | Interpreting clone state in experimentally measured GC selection

**A,** Experimental designs of the NP–OVA and RBD reporter studies. Division history, antigen-probe binding and GC-zone gates are measured experimental labels from separate sequencing libraries/cohorts. **B,** Cell UMAP for the independent NP–OVA day-14 output-sort study, coloured by measured dark-zone, light-zone, Myc-positive light-zone and plasma-cell gates. **C,D,** NP–OVA division-reporter clone map coloured by the fraction of cells in each family captured in the mCherry-low (at least six divisions) gate (**C**) or by mean V-region mutation frequency (**D**). **E,** RBD reporter cell UMAP coloured by measured mCherry gate. **F,G,** RBD reporter clone maps coloured by the mCherry-low fraction (**F**) or mean V-region mutation frequency (**G**). Points are reliable BCR-defined families; point size reflects captured family size. UMAP positions compare captured expression profiles and do not encode ancestry, temporal progression or fate.

## Figure 3 | Clonal co-occupancy of GC and output-like states in *Plasmodium*

**A,** Infection, sample preparation and treatment design. Early and late datasets are distinct cohorts; the late experiment includes saline and anti-malarial arms. **B,C,** Clone maps for early infection coloured by plasmablast (PB) occupancy (**B**) and later infection coloured by GC occupancy (**C**). **D,** Cell UMAP underlying the later clone map, coloured by annotated GC, PB and memory-like states. **E,** Predeclared examples of BCR-defined families containing captured GC and PB cells. Each inset is overlaid on the same cell UMAP; examples were selected by presence of both displayed states and captured family size. **F,** Per-mouse excess joint occupancy of GC–PB, GC–memory and memory–PB families relative to a within-mouse, clone-size- and isotype-preserving null. Dots are mice and horizontal lines are medians. These analyses establish same-mouse receptor-family co-occupancy of captured states, not GC re-entry, parent–offspring direction or future terminal fate.

## Figure 4 | Testing GC clone persistence and compartment bias in humans

**A,** Design of the longitudinal BNT162b2 lymph-node fine-needle aspiration and paired blood study. Sampling days are measured after first dose; repeated observations support persistence analyses only at the donor/family level. **B,** Human cell UMAP coloured by captured GC, lymph-node plasma-cell (LNPC), plasmablast (PB) and resting memory B-cell (RMB) states. **C,** Reliable Threadfin IGH family map coloured by the predominant captured state; black rings mark the 12 largest families carrying the author-provided Spike-positive clone label. Point size reflects capture count and coordinates represent profile similarity, not ancestry. **D,** Donor-stratified forest plot of odds that a reliable expression-programme family is labelled Spike-positive, comparing each programme with other families from the same donor. The input is the author’s clone-level `s_pos_clone` label propagated to sequence-defined Threadfin child families; it is an enrichment analysis, not a binding-affinity analysis. An asterisk marks programme resampling stability below 0.75; intervals resample families, not donors. **E,** Four predeclared large families with GC cells captured on at least three non-pooled dates. Stacked bars show the within-family distribution of captured states at each date; numbers are captured cells and NA denotes no captured cells. **F,** Clone-state retention across dates and between lymph node and blood, with 95% intervals. Retention describes repeated observational profiles and does not demonstrate memory re-entry into GC. **G,** Donor-balanced descriptive time course of V-region SHM in captured GC families at days 28, 35, 60, 110 and 201, stratified by author-labelled Spike-positive versus not identified as Spike-positive families. For each donor, the median family SHM is calculated within each timepoint and label; the plotted value is the equal-donor mean of these donor medians. Shading is the interquartile range across donors, not a confidence interval. Days 28 and 35 each include one non-pooled donor; the set of donors and families can differ by date. SHM is mutation history, not quantitative affinity; “not identified as Spike-positive” does not mean that every antibody was experimentally shown to be non-binding. **H,** Interpretation of the measurements: Threadfin families are sequence-defined, RNA profiles describe captured family states, the S+ call is the authors’ binding classification, and SHM is V-region mutation frequency. Their association does not make SHM or binary binding a measure of affinity.

## Figure 5 | What receptor-defined families add to the captured expression landscape

**A,** Biological anchors and the question contributed by each dataset class: reporter experiments, infection, longitudinal human vaccination, marrow/blood terminal-state sampling and exploratory datasets. **B,** Expression variance associated with clone identity for the analysed datasets, compared with within-library shuffles that preserve clone sizes and library composition. Coloured points are observed values; open points are shuffled means; black segments extend to the 95th percentile of the shuffled distribution. **C,** Numbers of expanded families (at least two captured cells) and reliable profiles (reliability at least 0.5). **D,** Within-family resemblance of predefined expression-module scores above matched background genes across the main datasets. **E,** Experimental resolution determines interpretation: reporter gates measure recent divisions, same-mouse data test co-occupancy, repeat donor samples test persistence and pure gates plus paired heavy/light receptors support receptor identity. No panel measures future fate, GC re-entry or binding affinity.

## Figure 6 | Comparing methods at the task they actually perform

**A,** Controlled reporter readout: common receptor-excluded RNA/BCR inputs,
native representations, fixed donor-private families and measured gate labels
held out by whole mouse. Representations are unsupervised/transductive; the
held-out labels enter only the downstream prediction test. **B,** Capabilities
and specialist strengths verified from original papers and official software;
these are documented tasks, not performance ranks. **C,** Absolute error in
captured family mCherry-low fraction, in percentage points, with one point per
label-held-out mouse and a bar at the median. Raw/donor-centred RNA means,
Threadfin mean/kernel profiles, full native Benisse geometry, BiGCN and clone2vec
are compared with the training-target-mean baseline. All readouts use the same
training-only kernel centring/scaling and nested whole-mouse ridge selection.
**D,** Expanded-family representation coverage and the common complete-family
intersection. Missing gate labels are excluded. This benchmark assesses captured
reporter composition; repertoire ancestry, affinity and future fate are separate
tasks. Final rendering requires both complete native dataset comparisons.

## Supplementary Figure 1 | Clone evidence and captured repertoire coverage

**A,** Distribution of captured cell counts per analysed family across the principal datasets. **B,** Number of families meeting the profile-reliability threshold (at least 0.5). **C,** Observed clone coherence and the corresponding within-library null. **D,** Sampling limits of the direct NP–OVA fate-sort dataset: its FACS labels support descriptive membership examples, whereas the small number of reliable profiles and non-significant coherence test preclude programme inference or validation of a fate predictor.

## Supplementary Figure 2 | Model-antigen clone relationships and orthogonal GC labels

**A,B,** Predeclared large NP–OVA (**A**) and RBD (**B**) families containing cells from both division gates, displayed on the shared cell UMAP for each experiment. **C,** RBD-protein cohort clone map coloured by the fraction of RBD-probe-positive cells. **D,** RBD-mRNA cohort clone map coloured by the fraction of dark-zone-sorted cells. **E,** Clone-state retention across division, RBD-binding and GC-zone gates, with 95% intervals. **F,** Directly measured NP–OVA output compartments in predeclared clone examples. **G,** Library-stratified association of division-gate occupancy with clone-profile variation in the NP–OVA reporter, RBD protein and RBD mRNA libraries. The orthogonal labels are separate experimental measurements; none establishes a lineage direction or an affinity value.

## Supplementary Figure 3 | *Plasmodium* sharing controls and candidate selection

**A,** Per-mouse time course of the fraction of GC-bearing families also captured in PB or memory states, shown separately for saline and anti-malarial arms. **B,** Predeclared GC–memory family examples. **C,** Observed GC–PB shared receptor groups under exact heavy-chain, exact paired heavy/light-chain and Threadfin-family definitions, shown without a sampling test. **D,** Supported and unsupported biological interpretations. **E,** Candidate selection and control rules. Shared receptor identity and captured state co-occupancy do not establish GC re-entry, parent–offspring relationships, future fate, antigen affinity or functional protection.

## Supplementary Figure 4 | GC and terminal-state biology are complementary records

**A,** Longitudinal composition of the human GC dataset by captured state. **B,** Verified single-donor marrow and blood library counts used for the marrow analysis. **C,** Median excess clone-associated variance for GC, cycling, light-zone, plasma-cell and memory expression modules across selected datasets. **D,** Sequence mutation history and current expression state provide different records: the calibrated reporter analysis did not detect a within-family association between SHM distance and expression distance. This null is limited to its tested cohorts and detectable effect sizes. **E,** Hierarchy of evidence from measured reporter/FACS labels through same-donor receptor membership and conditional controls to external functional tests.

## Supplementary Figure 5 | Non-GC validation across marrow and blood

**A,** Marrow/blood study design, including verified donor identity, tissue and sort gate. Pooled or mixed libraries are excluded from the primary analysis. **B,** Cell UMAP coloured by measured plasma-cell, memory-cell and combined sort gates. **C,** Reliable family map coloured by dominant measured pure sort gate. **D,** Descriptive expression signatures of marrow/blood programmes. **E,** Large predeclared families captured in both pure plasma-cell and memory-B-cell gates. **F,** Exact productive paired heavy- and light-chain receptor groups captured in both pure gates for donors 1681 and 1684. Shared receptors support common receptor identity across captured gates; they do not establish differentiation direction.

## Supplementary Figure 6 | Binding-label provenance and donor-level GC mutation trends

**A,** Crosswalk from author-defined clones to Threadfin IGH families. The histogram reports the number of Threadfin families per author clone. An author clone can be split into multiple sequence-defined families, whereas each Threadfin family maps to one author clone; propagated binding labels are therefore not newly measured biological replicates. **B,** Donor-specific composition of expression programmes by author-identified Spike-positive families. Each point is one donor/programme proportion among families with known labels. A FALSE call is not equivalent to a uniformly assayed antibody negative. **C,** The GC SHM comparison from Figure 4G with individual donor trajectories shown. Lines retain the equal-donor donor-median summary; the availability of donors and contributing families varies across dates. **D,** Expression-module signatures of the same captured human-family states. These scores reuse the expression data used to construct the profiles, so they describe composition and are not independent validation.

## Supplementary Figure 7 | Native benchmark sensitivity and execution scope

**A,B,** NP–OVA and RBD division-gate readout at thresholds of at least two or
five measured cells per family. Points and intervals are the mouse-level median
and IQR; stricter thresholds may retain different families and mice. **C,D,**
RBD antigen-probe and dark-zone sort gate fractions, calculated among measured
cells only, with whole-mouse held-out labels. **E,** Measured native model-stage
times on two CPU threads. PCA preprocessing and downstream readout are excluded;
Benisse additionally requires its pretrained encoder, which BiGCN also consumes.
Different stage scopes and allocations prevent an end-to-end package speed
ranking. Per-fold target prevalence, coverage, versions, ID hashes, peak command
RSS and the official-model adapters accompany the source tables. Training uses
one documented native configuration; BiGCN's upstream entry point has no fixed
seed, so the comparison does not quantify repeated-training variability.
