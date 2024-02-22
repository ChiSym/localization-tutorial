# TODO

* Expos
  * See "Prospects..." below
  * Rif comments: Correct understanding of initial pose.

* Debugging
  * Tweak the numerics parameters (including rejuvenation schedule / modifiers) to make the case more clearly
  * fix any text/comments, add any docstrings?, image filenames

* Visualization
  * BOTH notebook and slides
  * Better signposting of SMC logging code
  * Color sensor reading picture lines via when unexpectedly low likelihood.
  * plotting multiple traces: sequencing vs. tiling vs. alpha-blending (in each case indicate weights differently)
  * label all (hyper)parameters in visualizations

* Technical features
  * Low light: sensor coin flip / normal(max range value)
  * Otherwise hierarchical (sensor) model?
  * alternate world models for comparison with robot model.  Examples: variants of parameters



### Prospects for improving accuracy, robustness, and efficiency

Discussion of drawbacks.

One approach:
* Improve accuracy with more particles.
* Improve efficiency with smarter resamples (ESS, stratified...).
* Hope robustness is good enough.

Clearly not going to be fundamentally better than scaling a large NN, which is similar, just with offline training.

ProbComp advocates instead:
* A smart algorithm that fixes probable mistakes as it goes along.
* One idea: fix mistakes by running MH on each particle.  If MH changes them, then mistakes were fixed.
  * With generic Gaussian drift proposal.
  * An improvement: grid MH.
* Another idea: run SMCP3.
  * Get correct weights —>
    * algorithm has an estimate of its inference quality (math TBE: AIDE, EEVI papers)
    * higher quality resampling
* How good can we do, even with one particle?
  * Controller