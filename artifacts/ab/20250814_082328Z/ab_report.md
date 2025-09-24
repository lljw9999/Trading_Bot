# A/B Test Report: Policy vs. Baseline

**Generated:** 2025-08-14T08:23:28.066252Z
**Sample Size:** 5 aligned observations

## Primary Analysis (PnL)

### Result: **PASS**

- **Mean PnL Delta:** 0.0002
- **95% Confidence Interval:** [0.0002, 0.0002]
- **Effect Size (Cohen's d):** 9525857678563660.000

### Interpretation

✅ **Policy shows statistically significant improvement over baseline**
- Lower bound of 95% CI (0.0002) is positive
- Policy is expected to outperform baseline by 0.0002 units on average

## Secondary Metrics Analysis

### Entropy
✅ **BETTER** - Delta: 0.0940 ± 0.0049

### Q Spread
✅ **BETTER** - Delta: -0.1000 ± 0.0000


## Statistical Details

- **Bootstrap Resamples:** 1000
- **Confidence Level:** 95%
- **Alpha:** 0.05
- **Test Type:** Two-sided bootstrap CI

## Recommendations

- ✅ Policy demonstrates clear improvement - recommend for production deployment
- Monitor secondary metrics to ensure holistic performance improvement
- Consider gradual rollout with continued A/B monitoring