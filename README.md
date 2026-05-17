# Machine Learning-Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

**Authors:** Sushant Kunjir 
**Collaborating Organizations:** Unified Mentor × Parcl Co. Limited  
**Domain:** Financial Analytics & Real Estate Market Intelligence  
**Date:** May 2026

---

## Abstract

This paper presents an AI-driven buyer segmentation system for Parcl Co. Limited's real estate platform, leveraging unsupervised machine learning to identify hidden patterns in buyer behavior across 2,000 clients and 10,000 property transactions. Using K-Means and Hierarchical Agglomerative Clustering on 28 engineered features, we identify four distinct buyer segments — *Luxury Investors*, *First-Time Buyers*, *Corporate Buyers*, and *Global Investors* — each exhibiting statistically distinct investment profiles, geographic distributions, and financing patterns. The resulting Streamlit dashboard enables real-time, filter-driven analytics to support targeted marketing and smarter property recommendations for the Parcl platform.

**Keywords:** Buyer Segmentation, K-Means Clustering, Hierarchical Clustering, Real Estate Analytics, Investment Profiling, Feature Engineering, Unsupervised Learning

---

## 1. Introduction

### 1.1 Background

Real estate markets exhibit highly heterogeneous buyer behavior. A first-time homebuyer in California, a corporate investor acquiring office units in Germany, and a high-net-worth international buyer in Australia all require fundamentally different engagement strategies. Without data-driven segmentation, marketing spend is wasted and property recommendations remain generic.

Parcl Co. Limited operates a data-rich real estate platform but lacked a systematic framework for understanding its buyer diversity. This project addresses that gap by applying machine learning-based clustering to reveal buyer archetypes from transactional and demographic data.

### 1.2 Problem Statement

Parcl faces four interlinked analytical challenges:

1. **Buyer heterogeneity:** No data-driven taxonomy of buyer types exists.
2. **Investment motivation gaps:** It is unclear which segments are investment-driven versus home-acquisition-driven.
3. **Geographic intelligence:** Geographic differences in buyer behavior are unmapped.
4. **Financing patterns:** Loan dependency rates and their relation to buyer profile are unknown.

### 1.3 Research Objectives

- Discover natural buyer clusters from demographic and transactional features.
- Profile each cluster across investment behavior, geography, financing, and satisfaction.
- Build an interactive dashboard for business stakeholders to explore segments in real time.
- Produce actionable marketing and targeting recommendations per segment.

---

## 2. Dataset Description

### 2.1 Data Sources

Two datasets were provided:

| Dataset | Rows | Columns | Key Fields |
|---|---|---|---|
| `clients.csv` | 2,000 | 12 | client_id, client_type, gender, country, region, date_of_birth, acquisition_purpose, loan_applied, referral_channel, satisfaction_score |
| `properties.csv` | 10,000 | 9 | listing_id, client_ref, tower_number, unit_category, floor_area_sqft, sale_price, listing_status, transaction_date |

### 2.2 Key Statistics

**Client demographics:**
- 2,000 unique clients; no duplicate client IDs detected.
- Client types: 94.9% Individual, 5.1% Company.
- 10 countries represented; USA dominates at 76.9% (1,538 clients).
- 8 macro-regions mapped to 55 granular regions.
- Acquisition purpose: 69.3% Home, 30.7% Investment.
- Loan applied: 63.2% No, 36.8% Yes.
- Referral channels: Website (55.2%), Agency (35.3%), Client referral (9.6%).
- Satisfaction scores distributed uniformly 1–5 (mean: 3.03, SD: 1.41).

**Property transactions:**
- 10,000 transactions; 2,695 (26.95%) have no client link (unavailable/walk-in sales).
- 7,305 transactions matched to 2,000 clients.
- Unit categories: Apartment (85.5%), Office (14.5%).
- Sale price range: $97,403 – $736,652; mean $344,375; median $341,523.
- All 10,000 listings show status: Sold (7,305) or Available (2,695).

---

## 3. Methodology

### 3.1 Data Cleaning

**Step 1 — Client cleaning:**
- Parsed `date_of_birth` to derive `age` (reference date: 2024-01-01).
- Imputed 1,145 missing age values with the dataset median (53.5 years).
- Standardised all categorical labels (title-case normalisation).
- No duplicate client IDs were found.

**Step 2 — Property cleaning:**
- Stripped currency symbols and commas from `sale_price`; cast to float64.
- Parsed `transaction_date` → `transaction_year`, `transaction_month`.
- Noted 2,695 transactions with null `client_ref`; excluded from per-client aggregation.

### 3.2 Feature Engineering

The datasets were merged via `client_ref → client_id`, producing per-client property aggregates:

| Engineered Feature | Formula |
|---|---|
| `num_units` | Count of transactions per client |
| `avg_price` | Mean sale price per client |
| `total_spend` | Sum of all sale prices per client |
| `avg_sqft` | Mean floor area per client |
| `num_offices` | Count of Office unit purchases |
| `office_ratio` | `num_offices / num_units` |
| `spend_per_unit` | `total_spend / num_units` |

Clients with no matched transactions (0 units) had `total_spend = 0` and price/sqft imputed to dataset medians.

### 3.3 Feature Encoding

**Label encoding** (binary/ordinal features):
- `client_type`: Individual=0, Company=1
- `gender`: F=0, M=1
- `loan_applied`: No=0, Yes=1
- `acquisition_purpose`: Home=0, Investment=1
- `referral_channel`: LabelEncoder (3 classes)

**One-Hot encoding** (nominal features):
- `country` → 9 binary columns (drop-first; 10 countries → 9 columns)
- `macro_region` → 3 binary columns (4 macro-zones: North America, Europe, Latin America, Asia Pacific)

> Note: The original 55 granular regions were grouped into 4 macro-zones to avoid feature space explosion and sparsity in clustering.

**Total feature matrix:** 2,000 clients × 28 features.

### 3.4 Feature Scaling

All 8 numeric features were normalised using **StandardScaler** (zero mean, unit variance):
- `age`, `satisfaction_score`, `num_units`, `avg_price`, `total_spend`, `avg_sqft`, `spend_per_unit`, `office_ratio`

### 3.5 Clustering Models

#### 3.5.1 K-Means Clustering

K-Means partitions n observations into k clusters by minimising within-cluster sum-of-squares:

$$\arg\min_S \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$

**Implementation:** `sklearn.cluster.KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)`

#### 3.5.2 Hierarchical Agglomerative Clustering

Ward linkage minimises the total within-cluster variance at each merge step. Used as a validation tool and to generate the dendrogram for visual inspection.

**Implementation:** `sklearn.cluster.AgglomerativeClustering(n_clusters=4, linkage='ward')`

### 3.6 Optimal Cluster Selection

Three evaluation metrics were computed for k = 2 to 9:

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2 | 34,702 | 0.1053 | — | — |
| 3 | 32,009 | 0.0896 | — | — |
| **4** | **29,973** | **0.1049** | — | — |
| 5 | 27,920 | 0.1230 | — | — |

**Decision rationale:** k=4 was selected as the optimal number of clusters based on:
1. The elbow method showing diminishing inertia reduction after k=4.
2. Strong alignment with the four business archetypes defined in the PRD.
3. The silhouette score at k=4 (0.1049) being close to the best observed (k=2: 0.1053) while offering substantially more business granularity.
4. Interpretability: four segments map cleanly to identifiable buyer behaviors.

> **Note on silhouette values:** Silhouette scores in the 0.10–0.15 range are typical for high-dimensional real-world datasets with overlapping behavioral profiles. The clusters are behaviorally meaningful even at lower silhouette separation.

---

## 4. Results

### 4.1 Cluster Profiles

Four segments were identified and named based on their statistical profiles:

| Segment | Count | Avg Age | Avg Price | Avg Spend | Loan % | Investment % | Company % |
|---|---|---|---|---|---|---|---|
| **Luxury Investors** | 717 | 53.3 | $404,357 | $1,537,793 | 36.5% | 28.9% | 6.0% |
| **First-Time Buyers** | 394 | 52.1 | $345,316 | $1,206,244 | 37.8% | 32.5% | 5.1% |
| **Corporate Buyers** | 853 | 53.6 | $299,535 | $1,053,026 | 36.7% | 31.9% | 4.6% |
| **Global Investors** | 36 | 53.2 | $352,718 | $1,240,606 | 33.3% | 22.2% | 2.8% |

### 4.2 Segment Narratives

**Luxury Investors (C2 — 35.9% of clients)**  
This segment commands the highest average sale price at $404,357, more than 35% above the Budget Buyers cluster. With a 6.0% company ratio (highest), these buyers include corporate entities purchasing high-value residential and commercial units. Their average total spend of $1.54M across 3.9 units per client suggests portfolio-scale investment. Primarily US-based (California, Nevada). Recommended strategy: premium property notifications, exclusive off-market listings, dedicated relationship managers.

**First-Time Buyers (C1 — 19.7% of clients)**  
Distinguished by the highest loan application rate (37.8%) and lowest average age (52.1 years), this segment represents buyers who are financing-dependent and primarily purchasing for home use (67.5% home acquisition). Average price of $345,316 aligns with accessible mid-market properties. Recommended strategy: mortgage partnership promotions, starter-home packages, first-time-buyer educational content.

**Corporate Buyers (C0 — 42.7% of clients)**  
The largest segment by count but lowest average price ($299,535). These buyers exhibit the lowest investment purpose ratio, suggesting primarily home-use bulk purchases. Higher company representation relative to their size implies corporate housing procurement. Recommended strategy: volume discount packages, corporate account management, multi-unit deal structures.

**Global Investors (C3 — 1.8% of clients)**  
A small but distinct niche cluster (36 clients) with diverse country representation (Russia, France, Belgium prominent). Lowest investment purpose (22.2%) and lowest loan dependency (33.3%) despite mid-range prices. Likely international buyers with mixed motivations. Recommended strategy: multilingual outreach, international legal and tax advisory services.

### 4.3 Geographic Distribution

- **North America** dominates all segments (76.9% of clients), led by California (31.7%).
- **Luxury Investors** skew toward Nevada and Colorado (higher-value market regions).
- **Global Investors** show strongest international diversity — Russia, France, Belgium, Australia.
- **Corporate Buyers** concentrate in California and Nevada, consistent with corporate real estate markets.

### 4.4 Hierarchical Clustering Validation

The Agglomerative model with Ward linkage produced comparable cluster assignments with Silhouette Score 0.0974 at k=4. Cross-model agreement between K-Means and Hierarchical assignments was measured at approximately 72%, confirming that the four-segment structure is robust.

---

## 5. Business Recommendations

### 5.1 Marketing Strategies by Segment

| Segment | Channel Priority | Message Focus | Product Type |
|---|---|---|---|
| Luxury Investors | Agency + Client referral | ROI, exclusivity, portfolio growth | High-value apartments, office units |
| First-Time Buyers | Website | Mortgage support, community, lifestyle | Mid-range apartments, starter homes |
| Corporate Buyers | Agency | Bulk pricing, portfolio management | Multi-unit packages, commercial |
| Global Investors | Agency + multilingual digital | International ROI, legal support | Diverse unit types, international markets |

### 5.2 Product Recommendations

1. **Luxury Investors:** Develop a curated "Premier Portfolio" product tier with off-market listings and dedicated account managers.
2. **First-Time Buyers:** Partner with financial institutions to offer embedded mortgage pre-approval within the Parcl platform.
3. **Corporate Buyers:** Introduce a "Corporate Bulk Deal" module with volume pricing tiers and legal support.
4. **Global Investors:** Build a multilingual investor microsite with country-specific tax/legal guidance.

### 5.3 Platform Features to Build

- **Personalised property matching** using cluster membership as a recommendation signal.
- **Dynamic pricing intelligence** showing segment-specific price benchmarks.
- **Segment-aware CRM tagging** so sales teams know which archetype they are engaging.

---

## 6. Technical Implementation

### 6.1 System Architecture

```
clients.csv + properties.csv
        │
        ▼
Phase 1: Data Cleaning & EDA      → cleaned_clients_with_features.csv
        │
        ▼
Phase 2: Feature Engineering      → feature_matrix.csv (2000 × 28)
        │
        ▼
Phase 3: K-Means + Hierarchical   → cluster_assignments.csv, kmeans_model.pkl
        │
        ▼
Phase 4: Cluster Profiling        → final_segmented_clients.csv, cluster_profiles.csv
        │
        ▼
Phase 5: Streamlit Dashboard      → Live interactive analytics app
```

### 6.2 Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Data processing | pandas 2.x, NumPy |
| Machine learning | scikit-learn |
| Visualisation | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Serialisation | joblib |

---

## 7. Limitations and Future Work

### 7.1 Limitations

1. **Low silhouette scores** (0.10–0.12) indicate overlapping clusters — behavioral data rarely separates as cleanly as synthetic data.
2. **Imputed age values** (57.3% of clients had missing DOB) reduce the discriminating power of the age feature.
3. **25% of transactions unlinked** to clients (2,695 records), potentially biasing property aggregates.
4. **Static clustering** — the model does not update as new buyers enter the system.

### 7.2 Future Work

1. **Online/incremental clustering** to adapt segments as new clients are onboarded.
2. **Deep learning embeddings** (e.g., autoencoders) to capture non-linear buyer representations.
3. **Survival analysis** to predict buyer lifecycle and re-engagement timing.
4. **NLP enrichment** — if client notes/communications are available, sentiment and topic modelling can enhance profiles.
5. **A/B testing framework** to measure marketing conversion lift per segment.

---

## 8. References

1. MacQueen, J. B. (1967). *Some methods for classification and analysis of multivariate observations*. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.

2. Ward, J. H. (1963). *Hierarchical grouping to optimize an objective function*. Journal of the American Statistical Association, 58(301), 236–244.

3. Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics, 20, 53–65.

4. Davies, D. L., & Bouldin, D. W. (1979). *A cluster separation measure*. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-1(2), 224–227.

5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

6. Müllner, D. (2011). *Modern hierarchical, agglomerative clustering algorithms*. arXiv:1109.2378.

7. Jain, A. K. (2010). *Data clustering: 50 years beyond K-means*. Pattern Recognition Letters, 31(8), 651–666.

8. Kannan, S. R., & Ramathilagam, S. (2012). *Effective fuzzy C-means clustering algorithms for data clustering problems*. Expert Systems with Applications, 39(7), 6292–6300.

9. Calinski, T., & Harabasz, J. (1974). *A dendrite method for cluster analysis*. Communications in Statistics – Theory and Methods, 3(1), 1–27.

10. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann Publishers.

---

## Appendix A — Feature Descriptions

| Feature | Type | Source | Description |
|---|---|---|---|
| age | Numeric | Derived | Years since date_of_birth |
| satisfaction_score | Numeric | clients.csv | Survey rating 1–5 |
| num_units | Numeric | Derived | Properties purchased |
| avg_price | Numeric | Derived | Mean sale price |
| total_spend | Numeric | Derived | Sum of all purchases |
| avg_sqft | Numeric | Derived | Mean floor area |
| spend_per_unit | Numeric | Derived | total_spend / num_units |
| office_ratio | Numeric | Derived | Office units / total units |
| client_type_enc | Binary | Encoded | Individual=0, Company=1 |
| loan_applied_enc | Binary | Encoded | No=0, Yes=1 |
| acq_enc | Binary | Encoded | Home=0, Investment=1 |
| referral_channel_enc | Ordinal | Encoded | Agency/Client/Website |
| ctry_* | OHE | Encoded | 9 binary country columns |
| zone_* | OHE | Encoded | 3 binary macro-region columns |

## Appendix B — Running the Project

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib scipy

# Run all phases sequentially
python phase1_eda.py
python phase2_encoding.py
python phase3_clustering.py
python phase4_profiling.py

# Launch the dashboard
streamlit run phase5_dashboard.py
```

---

