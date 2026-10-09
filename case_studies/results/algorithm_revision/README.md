# Algorithm revision validation — 9 October 2026

These tables retain favourable and unfavourable results for the optional
reference state-distribution model introduced in Threadfin 4.1.0.

| Files | Experimental unit / interpretation |
| --- | --- |
| density_simulation* | 500 independent evaluation clones per regime; known regions, prior mismatch and pure-state stress tests |
| density_null_contrasts.csv | Conditional posterior interval flags under equal distributions; not a calibrated multiple-testing procedure |
| prior_calibration.csv | Reference-only split-cell choice of prior strength |
| density_larry* | Biological barcodes split between reference and evaluation; independent target cells, shared RNA coordinates |
| mouse_*_density_reporter* | Whole-mouse-held-out RNA preprocessing and readout; fixed settings, four methods, all thresholds |
| density_reporter_summary.csv | Complete method/threshold summaries with numbers of contributing mice and clones |
| representation_runtime.csv | Independent-process fitting time and peak RSS on identical synthetic inputs |
| case_integration_audit.json | Twelve corrected case studies; identifiers/capture/eligibility checked before preserving publication coordinates |
| final_validation.json | Final consistency and prediction-preservation checks |

The scripts are case_studies/validate_density_model.py,
validate_density_reporter.py and benchmark_representation_runtime.py.
The first accepts the optional argument larry; the reporter script accepts
mouse_np or mouse_rbd. Data paths use THREADFIN_DATA and the existing public-data
LARRY preprocessing cache. See reanalyse_clonotrace_larry.py for cache creation.
The integration helper accepts explicit --reruns and --backup paths and retains
the original backup on repeated invocation.

Posterior intervals are conditional and can severely under-cover; exact sampling
intervals assume independent multinomial draws conditional on fixed regions.
Shrinkage increases bias for pure/extreme clones. RNA means often win the
reporter endpoint. New density uses less memory than kernel in the reported
large example but takes longer. No official Clonotrace performance comparison
is included. These limitations are part of the results.
