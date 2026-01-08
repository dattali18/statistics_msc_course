---
title: "Statistics for Data Mining: Project Report"
author: "Daniel Attali, Sapir Bashan"
date: "Jan 2026"
---

# Statistics for Data Mining: Project Report

**Subject:** Statistical Analysis of the Wine Dataset
**Tools:** Python (Pandas, Num
Py, SciPy, Matplotlib)

---

## 1. Introduction & Data Description

### 1.1 Project Objective
The objective of this project is to apply statistical data mining techniques to a real-world dataset. The project is divided into three main parts:
1.  **Descriptive Statistics & Frequency Analysis:** Understanding the data distribution.
2.  **Estimation Theory:** Constructing confidence intervals for means and proportions.
3.  **Hypothesis Testing:** Performing parametric tests ($T$-test, ANOVA) and non-parametric tests ($\chi^2$) to draw inferences about the population.

### 1.2 The Dataset
We utilized the **Wine Dataset**, a classic classification dataset containing the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars (Classes 0, 1, and 2).

* **Total Instances:** 178
* **Attributes:** 13 continuous attributes (e.g., Alcohol, Malic Acid, Ash, Color Intensity) and 1 Target variable (Class).

---

## 2. Task 1: Basic Statistical Analysis

### 2.1 Descriptive Statistics
We performed an initial exploration of the numerical attributes to understand their central tendency and dispersion. As required, we calculated the Mean, Variance, Standard Deviation, Median, Range, and Midrange for all attributes.

* **See the attached file `results.xlsx` for the complete table of statistical metrics for all 13 attributes.**

### 2.2 Frequency Analysis
We selected **`malic_acid`** as the representative variable for detailed frequency analysis. The data was discretized into 10 bins to observe the distribution shape.

**Key Observations:**
* **Frequency Table:** The data was grouped into intervals (e.g., 0.734 - 1.246), showing absolute, relative, and cumulative frequencies.
* **Distribution:** The histogram indicates a right-skewed distribution, with the highest frequency of samples (59) falling in the `1.246 - 1.752` range.

![task_1_basic_statistical_analysis_13_0.png](task_1_basic_statistical_analysis_files/task_1_basic_statistical_analysis_13_0.png)
![task_1_basic_statistical_analysis_14_0.png](task_1_basic_statistical_analysis_files/task_1_basic_statistical_analysis_14_0.png)
![task_1_basic_statistical_analysis_15_0.png](task_1_basic_statistical_analysis_files/task_1_basic_statistical_analysis_15_0.png)
![task_1_basic_statistical_analysis_16_0.png](task_1_basic_statistical_analysis_files/task_1_basic_statistical_analysis_16_0.png)

The cumulative frequency analysis confirms that over 43% of the samples have a malic acid concentration below 1.752.

---

## 3. Task 2: Confidence Intervals (Estimation)

In this section, we estimated population parameters using Confidence Intervals (CI) at 90%, 95%, and 99% confidence levels. The **`alcohol`** attribute was used for the mean estimation.

### 3.1 Part 1: Mean Estimation with Known $\sigma$
* **Assumption:** We assumed the population standard deviation ($\sigma \approx 0.81$) is known (derived from the full dataset).
* **Sample:** A random sample of $n=32$ was taken.
* **Method:** $Z$-distribution.

**Results:**
The intervals indicate that as the confidence level increases, the interval width widens to capture the true mean with higher probability.
* **95% CI:** $(12.51, 13.08)$

### 3.2 Part 2: Mean Estimation with Unknown $\sigma$ (Small Sample)
* **Assumption:** Population $\sigma$ is unknown; Sample standard deviation ($s$) is used.
* **Sample:** A small random sample of $n=16$ was taken.
* **Method:** $t$-distribution ($df = 15$).

**Results:**
Due to the smaller sample size and the use of the $t$-distribution (which has heavier tails), the intervals are wider compared to Part 1.
* **95% CI:** $(12.78, 13.58)$
* **Interval Length:** $0.8014$

### 3.3 Part 3: Proportion Estimation
* **Variable:** Proportion of wines with **Alcohol content > 13.5**.
* **Sample:** $n=50$.
* **Observed Proportion ($\hat{p}$):** $0.32$ (16 successes out of 50).
* **Method:** Normal approximation for proportions.

**Results:**
* **95% CI:** $(0.1907, 0.4493)$
* We are 95% confident that the true proportion of wines with alcohol content over 13.5 lies between 19.1% and 44.9%.

![task_2_confidence_interval_sapir_12_0.png](task_2_confidence_interval_sapir_files/task_2_confidence_interval_sapir_12_0.png)
![task_2_confidence_interval_sapir_16_0.png](task_2_confidence_interval_sapir_files/task_2_confidence_interval_sapir_16_0.png)

---

## 4. Task 3: Hypothesis Testing (Two Groups)

We compared the means of different Wine Classes (Groups 0, 1, and 2) using the `alcohol` attribute.

### 4.1 Test for Difference in Means (Small Sample)
* **Comparison:** Group 0 vs. Group 1.
* **Sampling:** 6% of Group 0 ($n=4$) and 10% of Group 1 ($n=7$).
* **Hypothesis:** $H_0: \mu_0 = \mu_1$ vs. $H_1: \mu_0 \neq \mu_1$ (Two-tailed).
* **Test:** Independent $t$-test (Welch’s).

**Findings:**
* **$P$-value:** $0.0022$.
* **Conclusion:** We **Reject $H_0$** at all levels (90%, 95%, 99%). There is a statistically significant difference between the alcohol content of Group 0 and Group 1.

### 4.2 Test for Difference in Means (Large Sample, One-Tailed)
* **Comparison:** Group 0 vs. Group 2.
* **Sampling:** $n=32$ from each group.
* **Hypothesis:** $H_0: \mu_0 \le \mu_2$ vs. $H_1: \mu_0 > \mu_2$ (Right-tailed).

**Findings:**
* **$T$-statistic:** $5.0318$.
* **One-tailed $P$-value:** $\approx 0.0000$.
* **Conclusion:** We **Reject $H_0$**. The data strongly supports the hypothesis that Group 0 has a higher mean alcohol content than Group 2.

### 4.3 Test for Ratio of Variances ($F$-Test)
* **Objective:** To validate the assumption of equal variances used in standard $t$-tests.
* **Hypothesis:** $H_0: \sigma^2_A = \sigma^2_B$ vs. $H_1: \sigma^2_A \neq \sigma^2_B$.
* **Statistic:** $F = 0.8630$.

**Findings:**
* **$P$-value:** $0.6842$.
* **Conclusion:** We **Fail to Reject $H_0$**. There is no significant evidence that the variances differ.
* **Validity Check:** Since variances are statistically equal, a standard Student's $t$-test (pooled variance) would have been valid, though the Welch's test used in previous steps is robust regardless.

---

## 5. Task 4: Analysis of Variance (ANOVA)

We performed a One-Way ANOVA to test if the mean alcohol content differs simultaneously across all three groups (Class 0, 1, and 2).

* **Sampling:**
    * Group 0: 15% ($n=9$)
    * Group 1: 25% ($n=18$)
    * Group 2: 35% ($n=17$)
* **Calculations:**
    * $SS_{Between} = 16.1571$ (Variability due to Class difference)
    * $SS_{Within} = 10.1799$ (Random variability within Classes)
    * $SS_{Total} = 26.3371$

**Results:**
* **$F$-Statistic:** $32.54$.
* **$P$-value:** $\approx 0.0000$.
* **Conclusion:** We **Reject $H_0$** at 99% confidence. At least one wine class has a significantly different mean alcohol content than the others.

---

## 6. Task 5 & 6: Chi-Square ($\chi^2$) Tests

### 6.1 Goodness of Fit Test
* **Objective:** Test if the **`alcohol`** variable follows a **Normal Distribution**.
* **Method:** The data (30% sample) was binned into 5 equal-probability intervals based on theoretical normal quantiles.
* **Hypothesis:** $H_0$: Data is Normal vs. $H_1$: Data is not Normal.

**Findings:**
* **$\chi^2$ Statistic:** $2.7547$.
* **$P$-value:** $0.5997$.
* **Conclusion:** We **Fail to Reject $H_0$**. The sample data fits the Normal Distribution well.

### 6.2 Test for Independence
* **Objective:** Test for a relationship between **Wine Class (Target)** and **Color Intensity**.
* **Data Preparation:** `Color Intensity` was binned into ordinal categories: Low, Medium, High.
* **Contingency Table:** A cross-tabulation of Target vs. Color Category was created.

**Findings:**
* **$\chi^2$ Statistic:** $122.46$.
* **$P$-value:** $\approx 0.0000$.
* **Conclusion:** We **Reject $H_0$** decisively. There is a strong statistical dependence between the type of wine and its color intensity (e.g., Class 0 wines might consistently have higher color intensity compared to Class 1).

---

## 7. Conclusion
This project successfully applied statistical methods to the Wine dataset. We confirmed that the wine classes differ significantly in chemical properties (specifically Alcohol and Color Intensity). We also validated that the Alcohol attribute follows a normal distribution, justifying the use of parametric tests like the $T$-test and ANOVA.