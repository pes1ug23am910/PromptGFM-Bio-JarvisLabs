# Few shot learning for phenotype-driven diagnosis of patients with rare genetic diseases

_Text extracted from the PDF; images omitted._

## Page 1

                                    npj | digital medicine                                                                                                                               Article
                                    Published in partnership with Seoul National University Bundang Hospital

                                                                                                                                                     https://doi.org/10.1038/s41746-025-01749-1

                                    Few shot learning for phenotype-driven
                                    diagnosis of patients with rare genetic
                                    diseases
                                                                                                                                                                                   Check for updates
                                                        1,2,7                  1,3,7                       1                 1
                                    Emily Alsentzer , Michelle M. Li , Shilpa N. Kobren , Ayush Noori , Undiagnosed Diseases Network*,
                                    Isaac S. Kohane1 & Marinka Zitnik1,4,5,6

                                    There are over 7000 rare diseases, some affecting 3500 or fewer patients in the United States. Due to
                                    clinicians’ limited experience with such diseases and the heterogeneity of clinical presentations,
                  1234567890():,;
1234567890():,;

                                    ~70% of individuals seeking a diagnosis remain undiagnosed. Deep learning has demonstrated
                                    success in aiding the diagnosis of common diseases. However, existing approaches require labeled
                                    datasets with thousands of diagnosed patients per disease. We present SHEPHERD, a few-shot
                                    learning approach for multi-faceted rare disease diagnosis. SHEPHERD performs deep learning over a
                                    knowledge graph enriched with rare disease information and is trained on a dataset of simulated rare
                                    disease patients. We demonstrate SHEPHERD's effectiveness across diverse diagnostic tasks,
                                    performing causal gene discovery, retrieving “patients-like-me”, and characterizing novel disease
                                    presentations, using real-world cohorts from the Undiagnosed Diseases Network (N = 465), MyGene2
                                    (N = 146), and the Deciphering Developmental Disorders study (N = 1431). SHEPHERD demonstrates
                                    the potential of knowledge-grounded deep learning to accelerate rare disease diagnosis.

                                    Rare diseases affect 300–400 million people worldwide, yet each disease has        image10–15 or by comparing a patient’s set of phenotypic abnormalities to a
                                    a very low prevalence, involving no more than 50 per 100,000 individuals1–3.       knowledge base containing associations between phenotypes, genes, and
                                    Due to the low prevalence of rare diseases, most front-line clinicians lack        diseases16–20. Other approaches combine genotype and phenotype-based
                                    ﬁrsthand experience, resulting in numerous specialty referrals and expen-          approaches through Bayesian reasoning21,22 or by training machine learning
                                    sive clinical workups for patients across multiple years and institutions.         models to combine multiple handcrafted features23–28. Automated diagnosis
                                    Furthermore, patients with the same disease can present with variable              pipelines leveraging genotyping and phenotyping methods have improved
                                    symptoms, disease severity, and age of onset4. Such challenges make rare           diagnostic yields across a range of rare diseases29,30.
                                    disease diagnosis extremely difﬁcult; ~70% of individuals seeking a diag-               Advances in deep learning have considerably improved diagnostic
                                    nosis remain undiagnosed, and the genes underlying up to 50% of Men-               accuracy in other clinical areas31–40, achieving near-expert clinical accuracy
                                    delian conditions are unknown5,6. These diagnostic delays can lead to              for common diseases41–43. These methods offer several beneﬁts: they can
                                    redundant testing or unnecessary medical procedures, inappropriate or              automatically learn valuable features from patient cohort data and integrate
                                    delayed disease management, and irreversible disease progression if the time       multimodal phenotypic and genomic data into a shared feature space.
                                    window for intervention is missed.                                                 Despite the promise of deep learning approaches, however, there have been
                                         Machine-assisted diagnosis offers the opportunity to shorten diag-            limited attempts to leverage deep learning methods for diagnosing rare
                                    nostic delays for rare disease patients. Several strategies have been developed    genetic conditions due to the data scarcity problem. While deep learning
                                    to automatically analyze patients’ genetic and phenotypic data to aid diag-        approaches exist for image-based diagnosis10,11, there are limited approaches
                                    nosis. Genotype-based approaches focus on leveraging variant frequency             for phenotype-based diagnosis, and none automatically learn phenotypic
                                    and predicted pathogenicity to identify disease-causing variants7–9.               and/or genotypic features over patient phenotype terms23–28,44. Although
                                    Phenotype-based approaches prioritize genes by analyzing a patient’s facial        foundation models, such as large language models, have been leveraged for

                                    1
                                     Department of Biomedical Informatics, Harvard Medical School, Boston, MA, USA. 2Program in Health Sciences and Technology, Massachusetts Institute of
                                    Technology, Cambridge, MA, USA. 3Bioinformatics and Integrative Genomics Program, Harvard Medical School, Boston, MA, USA. 4Kempner Institute for the
                                    Study of Natural and Artiﬁcial Intelligence, Harvard University, Cambridge, MA, USA. 5Broad Institute of MIT and Harvard, Cambridge, MA, USA. 6Harvard Data
                                    Science Initiative, Cambridge, MA, USA. 7These authors contributed equally: Emily Alsentzer, Michelle M. Li. *A list of authors and their afﬁliations appears at the
                                    end of the paper.     e-mail: marinka@hms.harvard.edu

                                    npj Digital Medicine | (2025)8:380                                                                                                                                1

## Page 2

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                Article

clinical applications45,46, they are unable to achieve the diagnostic accuracy of   diseases to investigate the novel disease in depth. For each diagnostic task,
traditional rare-disease decision support tools47.                                  we illustrate SHEPHERD’s capabilities on patients in the Undiagnosed
      Training deep learning models requires high-quality labeled data with         Diseases Network and provide an interactive tool to explore SHEPHERD’s
thousands of diagnosed patients per disease. Yet, rare disease datasets are         predictions at https://huggingface.co/spaces/emilyalsentzer/SHEPHERD.
three orders of magnitude smaller. The low prevalence of rare diseases
makes it difﬁcult to obtain datasets of sufﬁcient size for deep learning, even      Results
with manual expert curation. Training deep learning models on datasets              Overview of the undiagnosed diseases network patient cohort
with smaller sample sizes can impact their generalizability. Due to each rare       We assemble a cohort of 465 patients in the Undiagnosed Diseases Network
disease’s heterogeneity and low prevalence, models are unlikely to have seen        (UDN) with molecular diagnoses. Most patients are diagnosed with a single
patients with the same (or similar) genetic disorders during training. Deep         causal gene that explains their symptoms; 14 patients (3%) have two causal
learning approaches for rare disease diagnosis must be able to extrapolate          genes, and two patients (0.4%) have three causal genes. Most patients in the
beyond the training distribution to novel genetic conditions and atypical           UDN receive an extensive clinical workup and whole genome or exome
disease presentations. Deep learning approaches that leverage medical               sequencing (Fig. 1a). Sequencing data is analyzed with the involvement of
knowledge are needed to overcome the limitations of traditional supervised          clinicians and genetic counselors to identify candidate genes that harbor
deep learning methods in this limited data setting.                                 variants likely to explain the patient’s symptoms. Once one to ﬁve strong
      Here, we introduce SHEPHERD, a deep learning approach for multi-              candidates are identiﬁed, causality is assessed by searching for genotype-
faceted diagnosis of patients with rare genetic conditions. SHEPHERD                and phenotype-matched individuals in human and animal databases or by
inputs patient phenotype terms and an optional list of candidate genes, and         introducing candidates into model organisms to determine in vivo impact50.
operates at multiple points throughout the rare disease diagnosis process to               Through this diagnostic process, patients are annotated with a set of
perform causal gene discovery, retrieve “patients-like-me” with similar             Human Phenotype Ontology (HPO) phenotype terms describing their
genetic and phenotypic features, and provide interpretable names for novel          clinical features and a set of candidate genes that may explain the patient’s
disease presentations. To overcome the limitations of supervised learning,          syndrome. Clinical experts additionally annotate diagnosed patients with an
SHEPHERD performs label-efﬁcient training by (1) training primarily on              Online Mendelian Inheritance in Man (OMIM) identiﬁer describing their
simulated rare disease patients and (2) incorporating medical knowledge of          disease (if available). Each patient is characterized by 23.9 HPO terms on
known phenotype, gene, and disease associations via knowledge-guided                average (SD = 16.1; Fig. 1c). The candidate genes are patient-speciﬁc and
deep learning. The simulated patients are created using an adaptive simu-           include genes in which the patient has a mutation. For each patient, the
lation approach that generates realistic rare disease patients with varying         diagnostic process creates two sets of candidate gene lists. The lists contain
numbers of phenotype terms and candidate genes48. Knowledge-guided                  genes considered at two different phases in the UDN diagnosis pipeline
learning is achieved by training a graph neural network to represent a              (Fig. 1a): VARIANT-FILTERED, a list produced by performing initial
patient (speciﬁcally, the patient’s presenting phenotypic features) about           variant-based ﬁltering of candidate genes, and EXPERT-CURATED, a list
other phenotypes, genes, and diseases. When a new patient arrives,                  including genes marked by clinical experts as strong candidates for the
SHEPHERD produces a mathematical representation (i.e., embedding) of                patient (Methods 3). The VARIANT-FILTERED gene lists are produced
the patient in the latent space such that the patient is embedded near the          using Exomiser24,51, a variant-based tool used in parallel to existing pipelines
patient’s causal gene(s), disease, and other patients with the same causal          at three UDN sites50. The two candidate gene lists contain 244.3 and 13.3
gene(s) or disease, and far from irrelevant genes and diseases and other            genes on average, respectively (SD = 244.0 and SD = 8.0; Fig. 1c). Each gene
patients with different diseases. Using SHEPHERD’s embedding space                  list is input to SHEPHERD to predict the causal gene (i.e., the gene har-
optimized for rare disease diagnosis, SHEPHERD can nominate genes and               boring variants that cause the patient’s disease) from both a long list of
diseases for every patient, even when no other patients are known to be             candidate genes derived from automated ﬁltering (i.e., VARIANT-FIL-
diagnosed with the same disease. Unlike existing methods, which rely on             TERED) and a short list of the strongest candidate genes that are more
handcrafted features, SHEPHERD learns patient representations informed              challenging to prioritize (i.e., EXPERT-CURATED).
by medical knowledge directly from patient phenotype terms to enable rare                  UDN patients have heterogeneous disease presentations: 378 unique
disease diagnosis with few (or zero) labeled examples.                              genes and 299 unique diseases are represented in the cohort, and 48% of
      We evaluate SHEPHERD on an external cohort of patients in the                 phenotype terms, 79% of genes, and 83% of diseases are represented in only
Undiagnosed Diseases Network (UDN)49, a nationwide initiative with 12               a single patient (Fig. 1d). This reinforces the need for machine learning
clinical sites in the United States tasked with diagnosing patients with rare,      models that can learn from sparsely labeled datasets. 11.4% of patients have
difﬁcult-to-diagnose genetic conditions. In addition to the multi-site UDN          a diagnosis in common with at least one other patient. Patients with the
cohort, our external evaluation includes a nationwide MyGene2 patient               same disease have only 67% of phenotype terms in common on average
cohort. SHEPHERD performs granular, phenotype-based causal gene dis-                (SD = 43%), and the closest shared ancestor (i.e., lowest common ancestor)
covery by ranking candidate genes from bioinformatics pipelines. We ﬁnd             in the Human Phenotype Ontology between their phenotype terms is 2.67
that SHEPHERD ranks the correct gene ﬁrst in 40% of patients spanning 16            hops away on average (SD = 0.81). In addition, 7% of patients have novel
disease areas, improving diagnostic efﬁciency by at least twofold compared          genetic diseases, and only 28% of each patient’s phenotypic features have
to a non-guided baseline. In addition, SHEPHERD nominates the correct               any known association with the causal gene on average (SD = 21%). The
diagnosis for patients with atypical presentations or novel diseases, ranking       assembled cohort of UDN patients has been evaluated at 12 clinical sites
the correct gene among the top ﬁve predictions for 77.8% of those hard-to-          across the United States (Fig. 1e). While 75.9% of patients are under 5 years
diagnose patients. SHEPHERD excels in diagnosing patients with novel                old, patients can present to the UDN with suspected genetic diseases in their
genetic conditions, ranking up to 86% of patients the same as or better than        40s or 50s (Fig. 1f). The cohort is predominantly White (80.6%) and non-
domain-speciﬁc approaches. By testing SHEPHERD on each disease area,                Hispanic (70.8%); smaller proportions of patients identify as Asian (9.2%),
clinical site, and year of diagnosis, we ﬁnd that SHEPHERD has sustained            Black or African American (4.5%), or other racial and ethnic backgrounds
performance over time and across diseases and clinical sites in the UDN.            (5.6%; Supplementary Fig. 1a, b). The sex distribution is relatively balanced,
Further, SHEPHERD generates patient representations that capture patient            with 47.7% male and 52.0% female patients (Supplementary Fig. 1c). Most
similarity (adjusted mutual information = 0.304) and enable retrieval of            patients present with neurological symptoms but can exhibit cardiac,
“patients-like-me” with similar genetic conditions. Finally, SHEPHERD can           musculoskeletal, rheumatic, and many other symptoms (Fig. 1g). Due to the
provide interpretable characterizations of novel disease presentations. By          lag between starting the process at the UDN and receiving the diagnosis,
describing novel diseases based on their similarity to known diseases,              most patients included in the analysis were evaluated by UDN clinicians in
SHEPHERD can point clinical researchers towards the most closely related            2016–2018 (Fig. 1h). The phenotypic heterogeneity and presence of novel

npj Digital Medicine | (2025)8:380                                                                                                                                2

## Page 3

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                           Article

Fig. 1 | Overview of SHEPHERD in the rare disease diagnosis pipeline. a After             consider a list of candidate genes (either variant-ﬁltered or expert-curated) or
years of failed diagnostic attempts, once a patient is accepted to the UDN, they          external patient cohort(s), depending on the prediction task of interest (e.g., causal
receive a thorough clinical workup and genetic sequencing, and their case is analyzed     gene discovery, patients-like-me identiﬁcation). For simplicity, the knowledge graph
in an iterative process to identify the candidate genes likely to explain the patient’s   is depicted using three shapes: circles as genes, squares as phenotypes, and pentagons
symptoms. SHEPHERD can be used throughout the diagnostic process: after the               as diseases; refer to Methods for all node types. c Number of HPO phenotype terms
clinical workup to ﬁnd similar patients, after the sequencing analysis to identify        and candidate genes in each of the two candidate gene lists across patients in our
strong candidate genes, and after the case review to further prioritize candidate         UDN cohort. d Overlap of phenotype terms, genes, and diseases across patients.
genes, characterize the patient’s disease, and/or validate candidate genes by ﬁnding      Most phenotype terms, genes, and diseases are found in only a single UDN patient.
phenotype- and genotype-matched patients. b SHEPHERD takes in as input the                e–h Number of patients in each e UDN clinical site, f age category, g primary pre-
patient’s set of phenotype terms and leverages an external rare disease knowledge         senting symptom, and h evaluation year. Figure adapted from images created in
graph to perform multi-faceted rare disease diagnosis. SHEPHERD can optionally            BioRender129–131.

and atypical diseases pose a challenge for diagnosis, requiring diagnostic                few (if any) labeled data points are available, is central to rare disease
technology that can accommodate previously unseen phenotypes, genes,                      diagnosis because of the low prevalence of each disease. Key to SHEP-
and diseases and leverage knowledge beyond direct gene, phenotype, and                    HERD’s ability to provide diagnostic prediction when zero or at most a few
disease associations (Supplementary Fig. 2). The UDN patients represent a                 labeled (diagnosed) patients per disease are available is to use a biomedical
diverse, independent cohort used exclusively for model evaluation.                        knowledge graph containing gene, phenotype, and disease relationships.
Importantly, these patients are not used to train SHEPHERD.                               SHEPHERD represents each patient as a set of phenotype terms from the
                                                                                          knowledge graph, which we refer to as a phenotype subgraph to emphasize
Overview of SHEPHERD algorithm                                                            that these terms are embedded within the graph’s structure (Methods 1). It
SHEPHERD takes a set of patient’s phenotype terms and candidate dis-                      leverages a graph neural network to jointly embed each patient’s phenotype
ease(s) or candidate gene(s) harboring causal variants as input, and per-                 subgraph and candidate genes or diseases into a latent representation space
forms multi-faceted diagnosis of the patient to identify causal genes, retrieve           such that the generated embeddings are informed by the structure of the
“patients-like-me” with the same causal gene or disease, and provide                      knowledge graph, and patients are embedded nearby their causal gene(s),
interpretable characterizations of novel disease presentations (Fig. 1b).                 disease(s), and other similar patients (Fig. 2a, b). Further, SHEPHERD uses
SHEPHERD can integrate into the rare disease diagnostic process workﬂow                   an attention mechanism to aggregate each patient’s phenotype terms to
at multiple points: (1) to ﬁnd similar patients after the patient’s clinical              generate a patient embedding. While not intended as a clinical interpret-
workup, (2) to identify strong candidate causal genes after the initial                   ability tool, the attention weights can be inspected post hoc to probe how the
sequencing analysis or in conjunction with the clinical case review, and (3)              model prioritizes different phenotypic features during training and
to characterize the patient’s disease and ﬁnd similar patients for experi-                inference.
mental or cohort validation after candidate causal genes are identiﬁed                          SHEPHERD is trained in a two-step process to learn embeddings of
(Fig. 1a, b).                                                                             biomedical concepts and patients with rare genetic diseases. First, SHEP-
      SHEPHERD is a few-shot geometric deep learning approach for rare                    HERD is pretrained via self-supervised learning to embed genes, pheno-
disease diagnosis. Few-shot learning, which can make predictions when very                types, and diseases by predicting the relationships (structure) of the

npj Digital Medicine | (2025)8:380                                                                                                                                            3

## Page 4

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                          Article

Fig. 2 | SHEPHERD architecture, training, and generalizability. a, b SHEPHERD           further trained on real-world patients (blue) and then evaluated on an independent
is trained in a two-step process. a First, the model is pretrained to embed the         cohort of real-world patients (green). Alternatively, SHEPHERD can directly be
biomedical knowledge in the knowledge graph. b Then, the pretrained model is            evaluated on real-world patients (green) without any additional training. d We
applied to the task of rare disease diagnosis. Patient information is overlaid on the   leverage real patient data derived from three distinct cohorts: the Undiagnosed
knowledge graph, and SHEPHERD generates an embedding for the patient phe-               Diseases Network (UDN; N = 465), MyGene2 (N = 146), and Deciphering Devel-
notype terms and each candidate gene, disease, or patient. The model is trained via a   opmental Disorders study (DDD; N = 1431). For simplicity, the KG is depicted using
loss function that encourages patient embeddings to be close to the embeddings of       three shapes: circles as genes, squares as phenotypes, and pentagons as diseases; refer
their causal gene or disease or other patients with the same causal gene or disease.    to Methods for all node types.
c SHEPHERD is trained on a large cohort of simulated patients (pink). It can be

biomedical knowledge graph (Fig. 2a; Methods 7). This step produces                     MyGene2, an online portal through which families with rare genetic con-
compact embeddings that can be adapted for a range of analyses and are                  ditions can share their health information to connect with clinicians and
generalizable by accounting for complementarity between diseases. Then,                 other patients53 (Methods 4); (3) a cohort of 1431 patients derived from the
using the pretrained model as initialization, SHEPHERD is trained for                   Deciphering Developmental Disorders study, an initiative from the United
multi-faceted diagnosis of patients with rare diseases via a novel objective            Kingdom and Ireland designed to diagnose patients with undiagnosed
function (Fig. 2b; Methods 7). We train SHEPHERD in a disease-stratiﬁed                 developmental disorders54 (Methods 5). Results are described in the fol-
manner (i.e., in which patients with the same disease are assigned either to            lowing sections.
the training or validation set, but not both) to enable SHEPHERD to gen-
eralize to diseases unseen during training.                                             SHEPHERD can perform causal gene discovery
      Due to the scarcity of data for patients with rare diseases, we leverage          A critical step in rare disease diagnosis is identifying the gene(s) that are
simulated but realistic rare disease patients for training SHEPHERD                     strong candidates for causing the patient’s syndrome (Fig. 1a). Given a
(Fig. 2c). We train SHEPHERD on a cohort of more than 40,000 synthetic                  patient’s set of phenotype terms and a list of genes in which the patient has a
rare disease patients representing over 2000 rare diseases in Orphanet                  mutation, SHEPHERD predicts genes that harbor variants most likely to
(Methods 6). There are 20 synthetic patients generated for each rare disease.           explain the patient’s presenting symptoms. SHEPHERD produces a score
The simulated patients were generated using an approach designed to                     for each candidate gene in the patient that fuses two complementary aspects
generate realistic rare disease patients grounded in medical knowledge, and             of information: an embedding-based metric that captures the global net-
they have been shown to phenotypically and genetically resemble real-world              work topology and a network-based metric computed using knowledge
rare disease patients48. The synthetic cohort is essential for training                 graph distance that captures local network information (Methods 11). We
SHEPHERD, as it is considerably larger, more diverse, and more repre-                   use SHEPHERD to prioritize genes found in both the EXPERT-CURATED
sentative of phenotype and genotype heterogeneity than any real-world                   and VARIANT-FILTERED candidate gene lists (Methods 3). In both
dataset of rare disease patients (Fig. 2c)48. This dataset, together with               instances, SHEPHERD performs granular prioritization by reﬁning lists of
knowledge-guided learning on the rare disease knowledge graph, enables                  patients’ candidate genes output by bioinformatics pipelines. For this ana-
deep learning for rare disease diagnosis. A notable byproduct of training the           lysis, we leverage patients from three cohorts: the simulated, MyGene2, and
model on synthetic data is that SHEPHERD’s model can be publicly released               DDD cohorts are used for training, and the UDN cohort is used for
without the risk of exposing patient information52. After training, SHEP-               validation.
HERD can be further trained on real-world patient cohorts or leveraged                        We report SHEPHERD’s performance in causal gene discovery as the
directly for rare disease diagnosis.                                                    average recall at k, deﬁned as the number of causal genes correctly predicted
      We leverage real patient data from three cohorts in this study (Fig. 2d):         in the top k ranked genes on average for all patients in the cohort. On the
(1) the UDN patient cohort (Methods 3); (2) a cohort of 146 patients from               EXPERT-CURATED gene lists, SHEPHERD ranks the patient’s causal gene

npj Digital Medicine | (2025)8:380                                                                                                                                           4

## Page 5

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                                                                                            Article

        a               SHEPHERD
                                                                               b                                                          c
                                                                                Site A
                                                                                                                                          Allergies / Immunology (N = 8)
                                                                                (N = 47)
                             Phrank                                             Site B                                                                    Cardiology (N = 9)
                            LIRICAL                                             (N = 41)
                                                                                Site C                                                               Endocrinology (N = 7)
                       ERIC (XRare)                                             (N = 37)
                                                                                Site D                                                            Gastroenterology (N = 11)
                              CADA
                                                                                (N = 41)
                                                                                                                                                   Musculoskeletal (N = 48)
                    LlaMa 3.1 – 70B                                             Site E
                                                                                (N = 37)                                                               Neurology (N = 148)
                     LlaMa 3.1 – 8B                                             Site F
        Mean shortest graph distance                                            (N = 44)                                                             Ophthalmology (N = 5)
                                                                                Site G
        Supervised graph embedding                                              (N = 40)
                                                                                                                                                                 Other (N = 48)
         Supervised PCA embedding
                                                                                                         Rank of causal gene                                                                   Rank of causal gene
                            Random

                                                Average recall at k

        d                                                                            e                                                                f
                                                                                                                    Expert-curated gene list                                Variant-filtered gene list
                        SHEPHERD
                                                                                     No gene-phenotype
                              CADA                                                                                0.82    0.79    0.59     0.79           0.80           0.80         0.62         0.57         0.58         0.53
                                                                                        associations
                      ERIC (XRare)                                                            (N = 40)

                           LIRICAL                                                     No gene-disease
                                                                                        associations              0.76    0.71    0.83     0.72           0.74           0.73         0.59         0.56         0.69         0.54
                 HiPhive (Exomiser)                                                           (N = 78)
                  PhenIX (Exomiser)
                                                                                     Novel disease gene           0.83    0.83    0.67     0.50           0.86           0.71         0.71         0.71         0.57         0.43
                             Phrank                                                           (N = 7)
                     LlaMa 3.1 – 8B
                                                                                           Novel disease          0.67    0.67    0.67     0.67           0.71           0.86         0.57         0.57         0.57         0.57
                    LlaMa 3.1 – 70B                                                           (N = 7)
        Mean shortest graph distance
         Supervised PCA embedding                                                                               ERIC LIRICAL CADA Phrank                  CADA PhenIX Phrank ERIC LIRICAL HiPhive
        Supervised graph embedding                                                                                                         SHEPHERD versus benchmark
                           Random

                                                                                                                                                                   1.0          0.8          0.6          0.4          0.2          0.0

                                                                                                                                                                                             Win Rate
                                                Average recall at k

Fig. 3 | SHEPHERD performs generalizable causal gene discovery. a Performance            SHEPHERD, six domain-speciﬁc approaches, ﬁve large language model, traditional
of SHEPHERD, four domain-speciﬁc approaches, ﬁve language model, traditional             machine learning, network science baselines, and a random baseline. The perfor-
machine learning, and network science baselines, and a random baseline. The              mance metric is average recall at k for k = 1, 5, 10, 25, and 50. Error bars denote
performance metric is average recall at k for k = 1, 3, and 5. Error bars denote         standard deviation over models trained with ﬁve random seeds. e, f Performance of
standard deviation over models trained with ﬁve random seeds. b, c Performance of        SHEPHERD against domain-speciﬁc algorithms in four extremely hard-to-diagnose
SHEPHERD in ranking causal genes stratiﬁed by b clinical sites and c primary             scenarios on e EXPERT-CURATED and f VARIANT-FILTERED gene lists. Shown
presenting symptoms. Each boxplot shows the median and interquartile range of the        is the win rate, the proportion of patients where SHEPHERD performs the same as or
rank of the causal gene. Whiskers extend to ±1.5 × IQR. d Performance of                 better than the benchmark algorithms.

ﬁrst in 40% of UDN patients, achieving a recall of 0.69 when k = 3 and 0.85              (LIRICAL21), shallow graph embeddings (CADA19), and information-
when k = 5 on average (Fig. 3a). On the much longer VARIANT-                             theoretic and random walk methods (HiPhive24). We further evaluate
FILTERED gene lists, SHEPHERD achieves an average recall of 0.21, 0.38,                  SHEPHERD against two large language models (LlaMa 3.1 8B and 70B55;
and 0.48 for k = 1, 5, and 10, respectively (Fig. 3d).                                   Supplementary Fig. 8). SHEPHERD performs comparably or signiﬁcantly
      We ﬁnd no signiﬁcant difference in performance across UDN sites                    better than all benchmarking approaches on the EXPERT-CURATED and
throughout the United States, patients with varying presenting symptoms,                 VARIANT-FILTERED gene lists (Fig. 3a, d and Supplementary Fig. 8).
and the year of evaluation by UDN clinicians (Fig. 3b, c and Supplementary               SHEPHERD outperforms the strongest domain-speciﬁc algorithms, LIRI-
Figs. 3a, 4a–c) on both the EXPERT-CURATED and VARIANT-                                  CAL and HiPhive, in prioritizing causal genes overall on both EXPERT-
FILTERED gene lists. These results indicate that SHEPHERD can gen-                       CURATED (p value = 4.27 × 10−2 for LIRICAL) and VARIANT-
eralize across clinical sites and diseases over time. Furthermore, we ﬁnd that           FILTERED (p value = 2.05 × 10−4 for LIRICAL and p value = 2.70 × 10−5
SHEPHERD’s performance does not correlate with the number of anno-                       for HiPhive) gene lists (Wilcoxon signed rank-sum test). SHEPHERD sig-
tated phenotype terms for each patient (Spearman’s ρ = 0.02 and ρ = −0.11                niﬁcantly outperforms the other domain-speciﬁc approaches in retrieving
for EXPERT-CURATED and VARIANT-FILTERED lists, respectively;                             the causal gene ﬁrst by up to 24.4% (p value = 4.92 × 10−16) and 7.7% (p
Supplementary Figs. 3c, 4e). Finally, we evaluate SHEPHERD’s perfor-                     value = 1.55 × 10−3) of patients on the EXPERT-CURATED and
mance as a function of the prevalence of the rare disease. We leverage the               VARIANT-FILTERED gene lists, respectively (McNemar’s test). Further-
number of submissions to ClinVar as a proxy for prevalence. We ﬁnd that                  more, SHEPHERD signiﬁcantly outperforms large language models in
SHEPHERD’s performance does not strongly correlate with the prevalence                   retrieving the causal gene ﬁrst by up to 20.1% (p value = 1.42 × 10−9) and
of the genetic condition (Spearman’s ρ = −0.17 and ρ = −0.16 for EXPERT-                 7.9% (p value = 6.85 × 10−3) of patients on the EXPERT-CURATED and
CURATED and VARIANT-FILTERED lists, respectively; Supplementary                          VARIANT-FILTERED gene lists, respectively, and the other machine
Figs. 3d, 4f). SHEPHERD’s ability to generalize represents an important                  learning approaches by up to 29.0% (p value 1.73 × 10−17) and 20.4% (p value
capability because rare disease patients are heterogeneous, and developing               1.44 × 10−15) of patients, respectively (McNemar’s test). For these statistical
separate predictive models that perform well for each patient subgroup is                tests, we apply Benjamin–Hochberg procedure for multiple testing
not feasible due to the low prevalence of the disorders.                                 correction.
      We evaluate SHEPHERD against 12 baseline approaches (Methods                             SHEPHERD’s strong performance demonstrates that SHEPHERD can
21). We select a network science algorithm and two supervised machine                    complement existing variant-based approaches for gene prioritization while
learning approaches as benchmarks to quantify the advantages of SHEP-                    leveraging the extensive knowledge sources of gene-phenotype associations.
HERD’s graph neural network approach. We also identify six domain-                       Using SHEPHERD, rare disease experts would need to evaluate 1026 genes
speciﬁc algorithms developed for causal gene discovery that leverage                     from the EXPERT-CURATED lists or 18,005 genes from the VARIANT-
information theory (Phrank16, PhenIX24, and ERIC25), likelihood ratios                   FILTERED lists to arrive at the causal gene for all 465 UDN patients. In

npj Digital Medicine | (2025)8:380                                                                                                                                                                                                          5

## Page 6

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                     Article

Fig. 4 | Causal gene discovery case studies for patients with novel genetic con-      FILTERED list are shown. The causal gene is highlighted in orange. The direct
ditions. SHEPHERD identiﬁes the causal gene even in atypical or novel disease         phenotypic neighbors of the causal gene are emphasized. In patient UDN-P1's
presentations. Each patient case study, shown in (a, b), includes the subset of the   network, the patient’s causal gene is directly connected to the disease in the
knowledge graph containing all nodes in the shortest path between the patient’s       knowledge graph. In patient UDN-P2's network, there is no disease node because the
phenotype terms, causal gene, and disease; a table of the patient’s phenotype terms   patient has a novel, uncharacterized syndrome. All panels, except those labeled as a
and attention weights learned by SHEPHERD; and bar plots of scores SHEPHERD           “patient card" (colored box with the information provided by the UDN), depict
assigned to each candidate gene in the EXPERT-CURATED and VARIANT-                    SHEPHERD's predictions or analyses performed on outputs of SHEPHERD.
FILTERED lists. The top and bottom ﬁve ranked genes in the VARIANT-

contrast, with non-guided ranking, experts would need to evaluate a total of          R2 = 0.102, Spearman’s ρ = 0.37 and R2 = 0.0004, Spearman’s ρ = 0.12 for the
2231 EXPERT-CURATED genes or 27,727 VARIANT-FILTERED genes,                           EXPERT-CURATED and VARIANT-FILTERED gene lists, respectively).
suggesting that SHEPHERD has the potential to improve diagnostic efﬁ-                        We evaluate SHEPHERD against the domain-speciﬁc models in four
ciency by 2.2-times and 1.5-times, respectively. Compared to the best                 hard-to-diagnose scenarios (Fig. 3e). We identify patients from the UDN
domain-speciﬁc approaches, LIRICAL and HiPhive, SHEPHERD reduces                      whose causal genes lack known associations with phenotype terms or dis-
the number of genes that experts need to consider by 97 (8.6%) and 5495               eases in the literature (based on our rare disease knowledge graph) and who
(23.3%) on the EXPERT-CURATED and VARIANT-FILTERED gene lists,                        have been identiﬁed by UDN experts as having novel disease genes or novel
respectively (LIRICAL), and by 1878 (9.4%) on the VARIANT-FILTERED                    diseases. SHEPHERD achieves win rates (i.e., ranks the causal gene the same
gene list (HiPhive).                                                                  or higher) of up to 82 and 83% for patients whose causal genes have no
                                                                                      known phenotype or disease associations, respectively, on the EXPERT-
SHEPHERD can diagnose patients with atypical and novel                                CURATED gene lists. On the VARIANT-FILTERED gene lists, the win
genetic diseases                                                                      rates are up to 80 and 74%, respectively. SHEPHERD achieves win rates of
Patients in the UDN have atypical or novel disease presentations, which               up to 67 and 83% for patients with a novel disease or novel disease gene,
makes them challenging to diagnose because there are no direct associations           respectively, according to UDN experts on the EXPERT-CURATED gene
between patients’ genes, symptoms, and the correct diagnosis. Conse-                  lists, and up to 86% on the VARIANT-FILTERED gene lists. The only
quently, the lack of direct linkage between patients’ phenotypic features and         subset of patients for which a baseline performs slightly better than
the correct diagnosis (causal genes) means that a lookup against medical              SHEPHERD consists of patients with novel disease genes, according to
knowledge bases is ineffective for diagnosis. We ﬁnd that SHEPHERD can                human experts in the UDN. In all other scenarios, SHEPHERD outperforms
identify the causal gene even when the patient’s presenting phenotypic                all baseline approaches, demonstrating SHEPHERD’s ability to diagnose
abnormalities are multiple hops away from the gene causing the disease in             patients with atypical and novel genetic diseases.
the knowledge graph. For 77.8% of patients whose phenotype terms are far                     We further demonstrate the use of SHEPHERD for patients diagnosed
away from their causal genes in the knowledge graph (i.e., more than two              with an atypical presentation of a known disease or a novel syndrome
hops away), SHEPHERD identiﬁes the correct causal gene among its top ﬁve              through two case studies on patients from the UDN. Patient UDN-P1
predictions from the EXPERT-CURATED gene list. No strong correlation                  (Fig. 4a; SHEPHERD Tool, Tab 1, Patient UDN-P1) received a diagnosis for
exists between SHEPHERD’s performance and the distance between the                    POLR3-related leukodystrophy three years after acceptance into the UDN.
patient’s phenotype terms and causal gene (Supplementary Figs. 3b, 4d;                While the involvement of gene POLR3A with leukodystrophy

npj Digital Medicine | (2025)8:380                                                                                                                                      6

## Page 7

https://doi.org/10.1038/s41746-025-01749-1                                                                                                              Article

(MIM:607694) is known, the patient’s case was challenging due to her              diseases and patients with odontologic and renal diseases cluster together in
atypical clinical presentation. Several of her presenting clinical features,      the embedding space. These clusters represent real co-occurrences of
including lack of tear production, premature adrenarche, laryngeal cleft,         symptoms in disease presentations. For instance, patients with odontologic
hearing loss, and high blood pressure, are not typical of leukodystrophy.         diseases, atypical dentin dysplasia, and orofaciodigital syndrome I, have both
Further, only 28.3% (13 out of 46) of the patient’s phenotype terms are           orofacial and renal disease presentations. Atypical dentin dysplasia is caused
directly linked to POLR3A in the knowledge graph, and the patient phe-            by a mutation in SMOC2, a matricellular protein involved in both cranio-
notype terms are 1.98 hops away from the causal gene in the knowledge             facial development and kidney ﬁbrosis57,58. Orofaciodigital syndrome I is
graph on average. The POLR3A gene is associated with ﬁve other                    caused by a mutation in OFD1, which is involved in organogenesis and plays
diseases, and 93.7% (192 out of 205) of phenotype terms directly linked to        a vital role in the normal growth of orofacial and kidney tissues59,60. These
POLR3A are not found in the patient, further complicating the diagnosis.          relationships reﬂect that diseases often involve multiple organ systems and
Despite this atypical disease presentation, SHEPHERD identiﬁes the                indicate that the embedding space can capture the relationship between
patient’s causal gene in the top 1 out of 17 and 86 candidate genes in the        patients with similar symptoms even when their diagnoses differ.
EXPERT-CURATED and VARIANT-FILTERED gene lists, respectively.
Strikingly, SHEPHERD can disambiguate diseases by optimally up- and               SHEPHERD can identify “patients-like-me” with similar genetic
down-weighting phenotypic features using an attention mechanism, and              diseases
correctly down-weights phenotypic features that are atypical of                   We next examine SHEPHERD’s ability to identify “patients-like-me” from a
leukodystrophy.                                                                   large cohort of rare disease patients. We either rank all simulated, UDN, and
     SHEPHERD can also identify strong candidate genes for patients with          MyGene2 patients (UDN-P3 and UDN-P4 cases) or all UDN and MyGene2
novel, uncharacterized syndromes. Patient UDN-P2 (Fig. 4b; SHEPHERD               patients (UDN-P5 and UDN-P6 cases; Fig. 5c; SHEPHERD Tool Tab 2) to
Tool Tab 1, Patient UDN-P2) was accepted into the UDN with congenital             identify patients most similar to the query UDN patient. We locate each
hypotonia and developmental delay. While no diagnosis was identiﬁed in            query patient and all similar patients with the same causal gene in SHEP-
the primary genomic and clinical evaluation, the patient was diagnosed            HERD’s embedding space, and ﬁnd that patients with the same causal gene
three years later with a novel PRKAR1B-related neurodevelopmental dis-            are embedded nearby. In all four patient cases, SHEPHERD retrieves
order. The PRKAR1B gene is not associated with known diseases. None of            patients with the same causal gene and disease as the query patient among
the 21 phenotype terms directly linked to PRKAR1B are found in the patient,       the top ﬁve predictions. Patients ranked above the patient with the same
and the average shortest path length from the patient’s phenotype terms to        causal gene have very similar disease presentations to the query patient. For
the causal gene is 2.4. Nevertheless, SHEPHERD identiﬁes the suspected            UDN-P4 and UDN-P5, the patients have a variant of the same disease
causal gene among the top 3 in the EXPERT-CURATED candidate list and              caused by a different gene (Fig. 5c). For UDN-P6, patients with Cofﬁn-Siris
the top 4 in the VARIANT-FILTERED candidate list, illustrating how                syndrome 8 (ranked ﬁrst) and GATAD2B-associated syndrome (ranked
SHEPHERD can assist in recognizing novel genetic diseases.                        second) both exhibit impaired intellectual development, hypotonia, feeding
                                                                                  difﬁculties, and hypertelorism, among other phenotypic abnormalities. For
SHEPHERD learns meaningful patient representations that                           UDN-P3, patients with X-linked intellectual disability due to GRIA3 (ranked
capture patient similarity                                                        ﬁrst) and Cofﬁn-Lowry syndrome (ranked second) share impaired intel-
Another critical consideration for rare disease diagnosis is ﬁnding patients      lectual development, seizures, scoliosis, and other phenotypic abnormalities.
that share the same disease or causal gene, commonly referred to as                    The most similar patients identiﬁed by SHEPHERD do not necessarily
“patients-like-me”56 (Fig. 1a). Starting from a set of patient phenotype terms,   have the most phenotype terms in common with the query patient. This
SHEPHERD ﬂags other patients in the cohort with similar genetic diseases          reﬂects SHEPHERD’s ability to capture phenotypic similarity rather than
suitable for follow-up diagnostic analysis. Concretely, SHEPHERD ﬁnds             just calculating a direct overlap in phenotype terms, typical of some
similar patients through a deep embedding scorer optimized to represent           information-theoretic approaches used in practice. In particular, patients
patients with the same causal genes or disease as nearby points in the            who share the same causal gene have two to four phenotype terms in
embedding space. For this analysis, we leverage patients from three cohorts:      common. Only 10.0, 9.0, 26.6, and 7.7% of the phenotype terms found in
the simulated cohort is used for training, and the UDN and MyGene2                query patients UDN-P3, UDN-P4, UDN-P5, and UDN-P6 are also found in
cohorts are used for validation.                                                  the most similar genotype-matched individual, respectively. In contrast,
       SHEPHERD represents each patient as a point in the embedding space         patients with the most phenotype terms in common with the query are
colored by the disease category of their diagnosed disease (Fig. 5). The          ranked at positions 366, 463, 41, and 16, respectively. For example, one
categories correspond to the 33 disease categories outlined in Orphanet           patient shares ten phenotype terms with UDN-P6, which is 38.5% of UDN-
(Methods 2). Robust clustering of patients by disease area (AMI = 0.304;          P6’s phenotypes, yet has a different causal gene and is ranked 16th. This
p value < 0.01) shows that SHEPHERD generates an embedding space that             capability of SHEPHERD to consider indirect, deep associations between
meaningfully captures patient relationships that can directly answer              genes and phenotypic features makes SHEPHERD highly complementary
“patients-like-me” queries. Remarkably, even though SHEPHERD is                   to graph-theoretic techniques and statistical tests that can only score direct
trained on simulated patients, it generalizes to real-world UDN and               associations, which can be ineffective for poorly characterized diseases.
MyGene2 cohorts, revealing disease-enriched regions in the embedding                   We next quantify SHEPHERD’s ability to identify “patients-like-me”
space where real-world patients are positioned nearby simulated patients          for each UDN patient from all patients in the real-world MyGene2 cohort.
with the same disease area (Supplementary Fig. 5).                                As before, we evaluate the average recall at k, here deﬁned as the number of
       To further evaluate patient embeddings, we compare embedding dis-          MyGene2 patients with the same causal gene as the query correctly pre-
tances between patients diagnosed with either the same or different disease       dicted in the top-k ranked patients on average for all UDN patients in the
(i.e., comparing diagonal vs. off-diagonal entries, Fig. 5b). We ﬁnd that         cohort. SHEPHERD ranks a patient with the same causal gene ﬁrst in 11.5%
distances between patients of the same category are signiﬁcantly smaller than     of UDN patients, achieving a recall of 0.31, 0.43, 0.49, and 0.53 for k = 5, 10,
between patients of different categories (p value <0.001 across all disease       25, and 50, respectively (Fig. 5a). We compare SHEPHERD to Phrank, an
categories; Mann–Whitney test with Benjamin–Hochberg procedure), which            alternative approach that can calculate phenotypic similarity. Phrank uses
indicates that SHEPHERD captures the similarity between patients with             information theory to calculate the similarity between two sets of phenotype
similar disease presentations. We also observe several distinct clusters of       terms based on shared ancestors in the Human Phenotype Ontology. We
disease categories in the embedding space (Fig. 5b and Supplementary Fig.         ﬁnd that SHEPHERD performs signiﬁcantly better than Phrank in identi-
7). For example, patients with neoplastic diseases and gastroenterologic          fying “patients-like-me” (Mann–Whitney p value = 0.04). SHEPHERD
diseases cluster together. Similarly, patients with hematologic and hepatic       ranks a patient with the same causal gene ﬁrst for 7.4% more patients and

npj Digital Medicine | (2025)8:380                                                                                                                              7

## Page 8

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                           Article

Fig. 5 | SHEPHERD identiﬁes patients-like-me from UDN and MyGene2                         space of all simulated (circle), UDN (up-facing triangle), and MyGene2 (down-
cohorts. a Performance of SHEPHERD in retrieving MyGene2 patients with the                facing triangle) patients colored by their Orphanet disease category. Each of the four
same causal gene as a UDN patient (n = 75 UDN patients with at least one matching         case studies consists of a zoomed-in UMAP displaying the query patient (star) and
patient in the MyGene2 cohort). SHEPHERD is benchmarked against Phrank, a                 all patients with the same causal gene as the query (colored circles) and a table
domain-speciﬁc algorithm. The performance metric is average recall at k for k = 1, 5,     containing information regarding the top ﬁve most similar patients retrieved by
10, 25, and 50. b Heatmap of the average distance between the phenotype embed-            SHEPHERD. Patients are bolded in the table if they share the same causal gene. All
dings of pairs of patients across disease categories. Darker colors indicate smaller      panels, except those labeled as a “patient card” (colored box with the information
distances and lighter colors indicate larger distances between patients of each pair of   provided by the UDN), depict SHEPHERD's predictions or analyses performed on
disease categories. c Two-dimensional UMAP plot of SHEPHERD's embedding                   outputs of SHEPHERD.

reduces the number of patients that clinicians need to consider by 703                    further strengthening the evidence that SHEPHERD can capture similarities
(17.2%) compared to Phrank.                                                               between different diseases with similar presenting symptoms, but can
      Finally, we evaluate whether SHEPHERD embeds patients with the                      nevertheless differentiate patients that have the same diagnosed disease.
same disease (rather than gene) closer to each other than to patients with
different diseases. Again, we compare UDN patients to MyGene2 patients.                   SHEPHERD provides an interpretable characterization of novel
We ﬁnd that embedding distances between patients diagnosed with the same                  diseases
disease are signiﬁcantly smaller compared to patients with different diseases             In addition to supporting causal gene discovery and patients-like-me
(p value = 2.42 × 10−8; Kolmogorov–Smirnov test; Supplementary Fig. 6),                   identiﬁcation, SHEPHERD can help characterize novel clinical

npj Digital Medicine | (2025)8:380                                                                                                                                            8

## Page 9

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                             Article

presentations through our current knowledge of rare diseases (Fig. 1a).                     analysis, we leverage patients from two cohorts: the simulated cohort is used
Given a patient’s set of HPO phenotype terms, SHEPHERD provides an                          for training, and the UDN cohort is used for validation.
interpretable summary of the patient’s disease based on its similarity to each                    We observe that SHEPHERD learns to embed patients near diseases of
disease in the KG. SHEPHERD produces a ranked list of all diseases using                    the same category; on average, 45.7% of the top ten ranked diseases with a
the embedding similarity between each disease and the patient’s phenotype                   known disease category belong to the same category as the patient’s disease,
terms, which are then summarized to generate a distribution of similarities                 which is nearly three times more than the random expectation alone
to disease categories. More concretely, SHEPHERD learns an embedding                        (16.4%). To evaluate SHEPHERD’s ability to provide interpretable disease
space in which the similarity between a patient and a disease is inversely                  names for patients with known rare diseases, we ﬁrst calculate the similarity
proportional to the embedding distance between the patient and their                        between UDN patients and all diseases. This allows us to assess whether the
diagnosed disease (Fig. 6a). Aggregating SHEPHERD-generated similarities                    patients are most similar to diseases that share the same disease category as
of individual diseases by their disease category enables interpretable char-                the patient’s disease (Fig. 6a). Concretely, for each patient, we stratify
acterization of the patient’s disease. For example, a patient’s presenting                  patients by their primary disease category and calculate the average simi-
syndrome may be w1% similar to rare neurologic diseases, w2% similar to                     larity of a patient to all disease nodes under each disease category. As
rare bone diseases, w3% similar to rare developmental defects during                        expected, we ﬁnd that patients tend to be most similar to diseases of the same
embryogenesis, etc. SHEPHERD can leverage gene-phenotype-disease                            disease category as their own. For example, patients with a rare bone disease
associations to generate granular descriptions of a patient’s disease. For this             are predicted to be most similar to diseases under the category of rare bone

Fig. 6 | SHEPHERD performs novel disease characterization. a Bar plots of the               four case studies contains: the percent similarity distributions of the patient’s phe-
similarity between UDN patients and diseases found in each disease category. We             notype terms to diseases in each disease category based on a phenotype search via the
group UDN patients by the disease category of their true disease and show plots for         KG (top) or SHEPHERD (bottom), a table of the ﬁve most similar diseases according
all categories with at least ﬁve patients. The bars that do not correspond to the disease   to SHEPHERD, and a table of the patient’s ﬁve phenotypic features that are most
category of each patient’s true disease are colored gray. b The column for each of the      highly attended by SHEPHERD.

npj Digital Medicine | (2025)8:380                                                                                                                                              9

## Page 10

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                Article

disease (13.0% similarity), followed by rare developmental defects during            are aligned with many of the patient’s symptoms, particularly duodenal
embryogenesis (10.2%), rare inborn errors of metabolism (9.6%), and rare             atresia, intestinal malrotation, pancreatic exocrine insufﬁciency, liver dis-
odontology diseases (8.2%). Similarly, patients with a disease categorized as        ease, and developmental delay. In contrast, the phenotype search approach
a rare developmental defect during embryogenesis, a rare inborn error of             predicts that the patient’s disease is most similar to diseases due to rare
metabolism, or a rare neurologic disease tend to be most similar to other            developmental defects during embryogenesis. Three of the top ﬁve most
diseases of the same category.                                                       similar individual diseases from SHEPHERD’s outputs—methylmalonic
      We examine two patients in depth to interrogate SHEPHERD’s pre-                acidemia with homocystinuria type cblF (MIM:277380; ranked by SHEP-
dictive capabilities for characterizing known rare diseases: UDN-P7 and              HERD as #1), neonatal hemochromatosis (MIM:231100; ranked by
UDN-P8. Patient UDN-P7 (Fig. 6b; SHEPHERD Tool Tab 3, Patient UDN-                   SHEPHERD as #2), and ALG8-CDG (MIM:608104; ranked by SHEP-
P7) received a diagnosis for limb-girdle muscular dystrophy 3 (sarcogly-             HERD as #4)—are also due to inborn errors of metabolism, and the diseases
canopathy; MIM:608099) due to variants in SGCA. SHEPHERD compares                    are associated with phenotypes that are similar to those seen in the patient,
the patient’s clinical presentation to diseases across 19 disease categories and     including abnormalities in liver and gastrointestinal function and devel-
ﬁnds that the patient is most similar to rare neurologic diseases, as expected.      opmental delay. Notably, the rare respiratory disease category is the third
From SHEPHERD’s predictions, two of the top ﬁve most similar diseases are            lowest-ranked category. UDN clinicians hypothesized that the patient’s
other types of AR limb-girdle muscular dystrophy, and all ﬁve are related to         GLYR1 variants cause a mislocalization of the cystic ﬁbrosis conductance
muscular dystrophy. We compare SHEPHERD to a simple phenotypic                       regulator (CFTR), which is associated with cystic ﬁbrosis. While the patient
search of the patient’s HPO terms to generate a distribution of similarities to      has gastrointestinal and pancreatic symptoms similar to those in cystic
disease categories. This phenotype search approach can correctly identify            ﬁbrosis, the patient does not have any of the pulmonary features classic for
the patient’s disease as a rare neurologic disease, but cannot produce disease-      that condition. Such granularity in SHEPHERD’s predictions is a reﬂection
level rankings. Patient UDN-P8 (Fig. 6b; SHEPHERD Tool Tab 3, Patient                of SHEPHERD’s ability to differentiate between diseases despite partially
UDN-P8) was diagnosed four years after acceptance to the UDN with the                overlapping phenotypes and causal genes sharing the same pathway.
bone disease spondyloepimetaphyseal dysplasia caused by a mutation in
RPL13. Again, SHEPHERD can ascertain that the patient’s symptoms are                 Discussion
similar to other bone diseases; all of the top ﬁve ranked disorders are rare         We present SHEPHERD, a deep learning approach for multi-faceted rare
bone diseases with overlapping phenotype terms found in the query patient.           disease diagnosis. SHEPHERD overcomes limitations of supervised deep
In contrast, the phenotype search approach does not identify UDN-P8’s                learning by (1) incorporating biomedical knowledge into the model via
disease as a rare bone disease; rather, it predicts that the patient has a disease   geometric deep learning on a knowledge graph, (2) leveraging label-efﬁcient
due to a rare developmental defect during embryogenesis. These ﬁndings on            learning to align patients with genes and phenotypes, and (3) training on a
our case studies of two patients with known rare diseases suggest that               large dataset of simulated rare disease patients in a disease-stratiﬁed manner.
SHEPHERD can produce correct and granular hypotheses about a patient’s               Further, the attention weights that are learned for generating phenotype-
rare disorder.                                                                       based patient embeddings can be inspected to provide insights into the
      We also investigate SHEPHERD’s hypotheses for two patients with                contribution of each phenotype term to the patient-speciﬁc prediction. As
novel genetic diseases, UDN-P9 and UDN-P10. UDN-P9 (Fig. 6b;                         shown in the evaluations on external multi-site patient cohorts with het-
SHEPHERD Tool Tab 3, Patient UDN-P9) was diagnosed with                              erogeneous disease presentations, SHEPHERD generalizes across pheno-
ATP5PO-related Leigh syndrome caused by a novel mutation in                          type terms, genes, and diseases and performs well on patients with
ATP5PO, a gene previously unassociated with any disease61. As Leigh                  heterogeneous clinical presentations and novel genetic conditions (Sup-
syndrome is a metabolic disorder with neuropathological features,                    plementary Fig. 2).
SHEPHERD correctly identiﬁes UDN-P9’s disease as most similar to                           A unique feature of SHEPHERD is its ability to generate clinico-genetic
diseases under the categories of rare inborn errors of metabolism and                representations of patients with rare genetic diseases. SHEPHERD repre-
rare neurological diseases. In contrast, the phenotype search method                 sents patient phenotype terms as subgraphs and candidate genes and dis-
incorrectly predicts a tie between a disorder due to a rare inborn error of          eases as nodes in the knowledge graph. The graph neural network then
metabolism and a rare neoplastic disease, failing to label the patient’s             generates embeddings by considering direct and indirect gene-phenotype-
disease as a neurological disorder. Three of the top ﬁve diseases—                   disease associations that are multiple hops away from each other in the
combined oxidative phosphorylation deﬁciency 39 (MIM:618397;                         knowledge graph. While many existing approaches rely exclusively on
ranked by SHEPHERD as #1), pyruvate dehydrogenase E3-binding                         known phenotype-gene-disease associations16,18, leveraging indirect asso-
protein deﬁciency (MIM:245349; ranked by SHEPHERD as #3), and                        ciations is essential for diagnosing patients with novel or atypical genetic
combined oxidative phosphorylation defect type 26 (MIM:616672;                       conditions (Supplementary Figs. 3b, 4d). Further, subgraphs provide a
ranked by SHEPHERD as #5)—are mitochondrial diseases affecting the                   ﬂexible mathematical deﬁnition for modeling sets of patient phenotype
same pathway as ATP5PO and result in a defect in the aerobic energy                  terms. Rather than modeling each phenotype term individually19, SHEP-
production. These diseases’ causal genes co-localize with ATP5PO62–65.               HERD encodes patients as a structured object (i.e., a subgraph) and con-
Combined oxidative phosphorylation deﬁciency 39 and combined oxi-                    siders the co-occurrence of phenotypic features when diagnosing rare
dative phosphorylation defect type 26 are associated with neurological               diseases. This joint modeling of patient phenotypic features is essential for
presentations of mitochondrial disease, including hypotonia, seizures,               identifying genetic mutations with pleiotropic effects74–76. This approach
and features of Leigh syndrome66. The remaining two most similar                     also helps mitigate variability in phenotype annotations by leveraging
diseases (ranked by SHEPHERD as #2 and #4) are rare neurologic                       relationships in the knowledge graph to connect patients described with
diseases with phenotype terms identical to UDN-P9’s. The causal gene,                different but related phenotype terms.
CNP, for the second-ranked disease, hypomyelinating leukodystropy-20                       SHEPHERD demonstrates that models trained on simulated patient
(MIM:619071), is three hops away from ATP5PO in the physical protein                 datasets apply to real-world clinical applications. While simulated data is
interaction network67,68, suggesting that they may be functionally                   increasingly used to augment training datasets for improving robustness
related69–71 or operate together72,73 to mediate phenotypic features asso-           and generalizability52,77–81, here we primarily use simulated patients to train
ciated with UDN-P9’s disease and hypomyelinating leukodystropy-20.                   SHEPHERD. Simulated data are not just an additional asset, but a critical
      Patient UDN-P10 (Fig. 6b; SHEPHERD Tool Tab 3, Patient UDN-                    necessity for training deep learning models to generate predictions on rare
P10), is characterized by SHEPHERD as most similar to diseases under the             diseases with scarce labeled diagnoses. The synthetic patients are generated
categories of rare inborn errors of metabolism, rare hepatic disease, rare           by a simulator48 based on clinico-genetic knowledge. Training on simulated
gastroenterological disease, and rare endocrine disease. These top categories        data mitigates concerns regarding privacy breaches, in which speciﬁc

npj Digital Medicine | (2025)8:380                                                                                                                               10

## Page 11

https://doi.org/10.1038/s41746-025-01749-1                                                                                                             Article

individuals can be identiﬁed from the training data82,83. Hence, a fully trained        SHEPHERD shows the utility and impact of deep learning for diag-
SHEPHERD model can be publicly released without privacy concerns.                  nosing rare disease patients. While other deep learning-powered diagnostic
      There are several extensions to this work. Our method relies on a            systems focus on common diseases for which large labeled datasets exist, this
knowledge graph of disease, gene, and phenotype associations. Other                study shows how deep learning can be used for rare diseases. The diagnostic
sources of information, such as variant-level information or databases of          process requires collaborations among bioinformaticians, clinicians, and
model organism phenotype-gene associations, can be incorporated as well84.         genetic counselors. Reviewing a single case can take many hours of a many-
SHEPHERD’s knowledge graph includes gene-phenotype-disease associa-                person team over days or weeks. SHEPHERD can substantially reduce the
tions and can be extended to include information from research literature26.       number of genes that human experts need to consider to provide a mole-
SHEPHERD’s phenotype-based approach can also be combined with                      cular diagnosis and identify patients with similar genetic conditions, even
variant-based prioritization approaches, such as those used in Exomiser24,         before they have undergone genetic sequencing. Deep learning-based
for even stronger causal gene discovery performance. The graph neural              diagnostic strategies like SHEPHERD create new opportunities to shorten
network underlying SHEPHERD can be extensible to multimodal data                   the diagnostic odyssey for rare diseases.
types. For example, gene co-expression data or textual descriptions of dis-
eases can be incorporated as node features. Given the importance of negative       Methods
phenotypes (i.e., explicitly absent symptoms) for differential diagnosis,          The Methods section is structured as follows: description of our rare disease
extending SHEPHERD to consider both present and absent symptoms may                knowledge graph; details of our rare disease patient cohorts; formulations of
improve patient representations for multi-faceted rare disease diagnosis.          SHEPHERD, our algorithmic approach for rare disease diagnosis; details
Incorporating temporal modeling of phenotype progression into the patient          regarding model training; and descriptions of our statistical analysis and
simulation process and SHEPHERD’s framework could further enhance its              evaluation setup.
ability to recognize age-dependent disease manifestations and improve
diagnostic accuracy across the lifespan. It is also worth exploring various        Rare disease knowledge graph construction
graph transformer architectures to potentially improve SHEPHERD’s                  We create a comprehensive knowledge graph (KG) for rare disease diag-
ability to harness the beneﬁts of global attention while preserving the graph’s    nosis. We start with PrimeKG88 and adapt it to the rare disease diagnostic
structural nuances. Recent large language model (LLM) approaches for rare          setting by removing drug-speciﬁc entities and relations and incorporating
diseases could complement SHEPHERD by leveraging biomedical pre-                   additional sources of gene, phenotype, and disease relationships. The
training and retrieval-augmented generation to support diagnosis from              resulting rare disease KG contains seven node types (i.e., phenotype, protein,
unstructured clinical notes and literature without requiring structured            disease, pathway, molecular function (MF), cellular component (CC), and
phenotype terms85–87. However, out-of-the-box LLMs have been shown to              biological process (BP)) and 15 unique relation types (i.e.,
underperform in rare disease diagnosis, likely due to the rarity of many           phenotype–protein, disease–phenotype(-) (indicating that disease does not
conditions in biomedical corpora. Hybrid models that either incorporate            have phenotype), disease–phenotype(+) (indicating that disease has phe-
graph-based representations into LLMs or integrate LLM-derived embed-              notype), protein–pathway, disease–protein, protein–MF, protein–CC,
dings into structured knowledge graphs may help mitigate these limitations,        protein–BP, BP–BP, MF–MF, CC–CC, phenotype–phenotype,
enhancing diagnostic accuracy and interpretability. Evaluating the utility of      protein–protein, disease–disease, pathway–pathway).
SHEPHERD in prospective clinical workﬂows represents an important next                   Relationships are extracted from the following data sources: Gene
step to assess its performance in practice and understand its impact on            Ontology (GO)90, Reactome pathway knowledgebase91, DisGeNET92,
clinical decision-making. Finally, while efforts like the UDN are critical for     NCBI93, Human Phenotype Ontology (HPO)94, MONDO disease
establishing diagnoses for rare disease patients, they alone cannot address        ontology95, and Orphanet96. PrimeKG contains disease–protein relation-
the rare disease burden. Approaches like SHEPHERD can help identify and            ships from DisGeNET, and we include additional disease–protein and
diagnose rare disease patients using claims data, electronic health records,       disease–phenotype relationships from Orphanet if they are not already
and other data types. SHEPHERD’s ability to characterize a patient’s clinical      present in the KG. All phenotype terms are mapped to the Human Phe-
presentation can be used to identify sub-specialists who should review the         notype Ontology, all genes/proteins are mapped to Ensembl identiﬁers, and
patient’s case for the diagnostic recommendation.                                  all diseases are mapped to MONDO identiﬁers. When a concept is repre-
      Our study has a few limitations. First, continually updating the             sented in the HPO and MONDO ontologies, we remove the MONDO
knowledge graph with gene-phenotype-disease associations can improve               identiﬁer. This differs from the original PrimeKG preprocessing, where
SHEPHERD’s performance. To this end, the knowledge graph curation and              conﬂicting identiﬁers are mapped to MONDO IDs. We perform all other
processing approaches are fully reproducible, and the graph can be auto-           preprocessing as in the original PrimeKG knowledge graph. In particular,
matically updated as new data becomes available88. Second, the still-              duplicate and self-loop edges are removed, and only the largest connected
undiagnosed UDN patients may be more challenging than the already-                 component of the graph is retained to ensure connectivity. For additional
diagnosed ones SHEPHERD was tested on. There are two categories of still-          information about each data source and the harmonization process, refer to
undiagnosed patients: patients admitted to the UDN years ago who have yet          PrimeKG88.
to receive a diagnosis due to sequencing limitations (e.g., hard-to-detect               We enforce homophily between genes and phenotypes by computing
variant types such as short tandem repeats or structural variants, missing         the triadic closure between gene–disease and disease–phenotype edges97,98.
second variants in recessive disorders, variants that lie in difﬁcult-to-          We extract the largest connected component to ensure the KG is fully
sequence regions or are masked due to biases in the human reference                connected. The largest connected component retains 99.91% of the nodes
genome and ancestral genomes89), and patients recently admitted to the             and 99.99% of the edges from the knowledge graph. Finally, we add reverse
UDN. SHEPHERD can be evaluated on the still-undiagnosed patients                   edges to ensure the KG is represented as an undirected graph during model
whose causal variants will be detectable by deep whole-genome sequencing.          training.
The lack of an observed drop in SHEPHERD’s performance for recently                      The ﬁnal knowledge graph contains 105,220 nodes and 1,095,469
diagnosed patients suggests that data leakage (i.e., information about older       edges. Tables 1, 2 outline the number of nodes and edges by node type and
diagnoses being incorporated into the knowledge graph) has not occurred,           relation type, respectively.
evidently avoiding the bias that would otherwise cause overﬁtting of the
model to the training data. Finally, like many genomic studies, our datasets       Rare disease patient cohorts
likely overrepresent individuals of European descent. This demographic             We use four distinct rare disease patient cohorts for training and evaluating
bias is an important limitation that could affect generalizability, and high-      SHEPHERD: UDN (section “Patients in the Undiagnosed Diseases Net-
lights the need for more inclusive rare disease datasets in future work.           work”), a real-world cohort of hard-to-diagnose patients in the

npj Digital Medicine | (2025)8:380                                                                                                                            11

## Page 12

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                          Article

Table 1 | Statistics about nodes in the rare disease                                         to bring together clinical and research experts around the United States to
knowledge graph                                                                              diagnose patients with rare genetic conditions99. The Undiagnosed Diseases
                                                                                             Network study is approved by the National Institutes of Health institutional
 Node type          Count         Average degree           Vocabulary
                                                                                             review board (IRB), which serves as the central IRB for the study (IRB
 Phenotype          15,874        49.5 ± 190.6             Human phenotype ontology94        Protocol 15HG0130). All patients accepted to the UDN provide written
 Disease            21,233        17.1 ± 37.4              Mondo95                           informed consent to share their data across the UDN as part of a network-
 Protein            21,610        74.2 ± 119.8             Ensembl125                        wide informed consent process.
 Pathway            2516          19.0 ± 29.9              Reactome91
                                                                                                    The UDN consists of 12 clinical sites across the United States that
                                                                                             evaluate patients, a sequencing core, a model organism screening center, a
 MF                 11,169        8.7 ± 127.4              Gene ontology90
                                                                                             central biorepository, a metabolomics core, and a coordinating center.
 CC                 4176          22.3 ± 197.2             Gene ontology90                   Patients are admitted to the UDN if they have objective ﬁndings, and clinical
 BP                 28,642        8.7 ± 28.1               Gene ontology90                   testing has failed to produce a diagnosis. Most admitted patients receive
Reported is the number of nodes by node type.                                                exome or whole genome sequencing and an extensive clinical workup.
MF molecular function, CC cellular component, BP biological process.                                We include patients from the Undiagnosed Diseases Network who meet
                                                                                             the following criteria: (1) at least one phenotype term describing their clinical
                                                                                             presentation, (2) at least ﬁve candidate genes potentially explaining their
                                                                                             symptoms, and (3) a diagnosis classiﬁed as “certain” or “highly likely” based
Table 2 | Statistics about the edges in the rare disease
                                                                                             on the UDN’s diagnostic certainty annotations29. Diagnoses with minimal
knowledge graph
                                                                                             uncertainty are considered “certain,” while those with some uncertainty–yet
 Relation type                 Count         Sources                                         still sufﬁcient for clinical decision-making–are classiﬁed as “highly likely.”
 Phenotype–phenotype           21,925        HPO94                                                  We construct patient subgraphs using the phenotypes obtained
                                                                                             through deep phenotyping. Deep phenotyping of patients during the clinical
 Phenotype–protein             10,518        HPO94, Mondo95, Orphanet96
                                                                                             workup is a central component of the UDN process. Clinicians annotate
 Disease–disease               35,167        Disgenet92, Mondo95, Orphanet96                 each patient with a set of terms from the Human Phenotype Ontology
                        (−)
 Disease–phenotype             1483          HPO94, Disgenet92, Mondo95, Orphanet96          (HPO) using PhenoTips, a tool integrated into the electronic health record
 Disease–phenotype      (+)
                               204,779       HPO94, Disgenet92, Mondo95, Orphanet96          that allows for structured phenotyping of patient symptoms100. We map all
 Disease–protein               86,299        Disgenet92, Mondo95, Orphanet96
                                                                                             phenotype terms to the same version of the Human Phenotype Ontology
                                                                                             (v2019), discard 406 unique prenatal phenotype terms related to the
 Protein–protein               321,075       Menche et al.126, Biogrid127, String128, Luck
                                             et al.73
                                                                                             mother’s pregnancy and use all remaining phenotype terms to construct
                                                                                             patient subgraphs. Each patient subgraph is formed from the phenotype
 Protein–pathway               42,646        Reactome91
                                                                                             nodes in the rare disease knowledge graph that describe the patient’s
 Pathway–pathway               2535          Reactome91                                      symptoms (Methods 8). We construct phenotype subgraphs for the 465
 Protein–MF                    69,530        Gene ontology90                                 UDN patients with annotated phenotype terms who have received a
 Protein–CC                    83,402        Gene ontology90                                 molecular diagnosis as of January 5, 2022.
                                                                                                    We obtain EXPERT-CURATED candidate gene lists from the UDN.
 Protein–BP                    144,805       Gene ontology90
                                                                                             Genomic samples for each patient are sequenced at Baylor Genetics or
 MF–MF                         13,574        Gene ontology90
                                                                                             Hudson Alpha. All candidate genes are standardized to Ensembl gene
 CC–CC                         4845          Gene ontology90                                 identiﬁers. We construct an EXPERT-CURATED candidate gene list for
 BP–BP                         52,886        Gene ontology90                                 each patient from the patient’s sequencing data. Importantly, these gene lists
Reported is the number of edges by relation type.                                            are unique to each patient. The EXPERT-CURATED candidate gene list for
HPO human phenotype ontology.                                                                each patient includes the union of both (1) disease-associated and other
                                                                                             clinically-relevant genes listed on the patient’s clinical sequencing reports
                                                                                             from Baylor or Hudson Alpha per the UDN protocol and American College
                                                                                             of Medical Genetics and Genomics (ACMG) guidelines29,101,102 and (2) genes
Undiagnosed Diseases Network, MyGene2 (section “Patients in the                              that were prioritized by UDN clinical teams who handled the patient’s case.
MyGene2 portal”), a publicly available real-world cohort of patients with                    The genes in this list represent the strongest candidates identiﬁed by the
rare genetic conditions who have opted to share their information on the                     UDN sequencing core or the clinical team. In addition, the list often includes
MyGene2 Portal, DDD (section “Patients derived from the Deciphering                          known disease-causing genes, genes with suspected pathogenic variants, or
Developmental Disorders study”), a publicly available aggregated summary                     genes expressed in tissues relevant to the patient’s clinical presentation.
of real pediatric patients with severe developmental disorders in the Deci-                  While the EXPERT-CURATED gene list contains the strongest candidates,
phering Developmental Disorders study, and SIMULATED (section                                the list nevertheless requires further ﬁltering to identify the ultimate causal
“Simulated patients with rare Mendelian disorders”), a large diverse and                     gene(s) that explain the patient’s condition. We exclude patients whose
realistic simulated patient cohort representing 2,134 unique rare diseases in                candidate gene lists have fewer than ﬁve candidate genes for the causal gene
Orphanet. The diseases found in each cohort are in Supplementary Table 1.                    discovery task. The cohort contains 278 patients with at least ﬁve EXPERT-
      For every patient cohort, we categorize each patient’s causal disease                  CURATED candidate genes.
according to the 33 disease categories outlined in Orphanet. We map all                             We obtain VARIANT-FILTERED candidate gene lists from the UDN.
diseases to Orphanet and leverage the Orphanet linearisation process                         As part of the UDN analysis pipeline, the UDN performs the whole genome
(http://www.orphadata.org/cgi-bin/rare_free.html) to assign each disease to                  and exome sequencing for a patient and their immediate family members.
a single disease category based on a series of rules that consider the most                  Here, we use the patients’ whole genome sequencing (WGS) data, which are
severely affected body system and the specialists most likely to be involved in              aligned to the GRCh38.p13/hg38 human genome build and have undergone
treatment.                                                                                   variant calling via the Genome Analysis Toolkit (GATK) best practices
                                                                                             workﬂow50. Please refer to ref. 50 for more details about the computational
Patients in the Undiagnosed Diseases Network                                                 workﬂow across UDN sites. Access to the UDN patients’ WGS data allows us
The Undiagnosed Diseases Network (UDN) is a nationwide research study                        to construct for each patient a VARIANT-FILTERED candidate gene list
supported by the National Institutes of Health Common Fund, which aims                       consisting of genes that have at least one variant and that have been prioritized

npj Digital Medicine | (2025)8:380                                                                                                                                         12

## Page 13

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                 Article

by a variant-level prioritization algorithm. We leverage the Exomiser algo-          Disorders study. This initiative recruited nearly 14,000 children with severe,
rithm, which considers variant frequency, predicted pathogenicity, and (if           undiagnosed developmental disorders from the United Kingdom and
family members’ sequencing data are available) mode of inheritance103. While         Ireland54. Among enrolled probands, 42% were female, 16% were of non-
Exomiser can leverage known associations between genes and phenotypic                European ancestry, and the median age at recruitment was 7 years.
features, we do not use it to construct our VARIANT-FILTERED candidate                     We retrieved data containing the sets of phenotype terms and asso-
gene lists. We analyze the patients’ variant-called WGS data (i.e., variant call     ciated genes from DECIPHER (https://www.deciphergenomics.org/ddd/
format, or VCF) using Exomiser under the following inheritance modes:                ddgenes) on May 10, 2023. We remove genes where the evidence supporting
autosomal dominant, autosomal recessive homozygous alternate, autosomal              a causal role for the gene is either limited or moderate, and we use the
recessive compound heterozygous, X dominant, X recessive homozygous                  remaining genes and associated phenotype term sets to construct the patient
alternative, X recessive compound heterozygous, and mitochondrial. Their             cohort. We map all genes to Ensembl identiﬁers and diseases to MONDO
respective cutoff values (i.e., the maximum minor allele frequency in percent        identiﬁers, and build patient subgraphs from the set of HPO terms asso-
(%) permitted for an allele to be considered a causative candidate under that        ciated with each patient. Non-causal candidate genes for each patient are
mode of inheritance) are 0.1, 0.1, 2.0, 0.1, 0.1, 2.0, and 0.2. We remove variants   constructed by sampling genes in the knowledge graph neighboring the
with non-coding effects (i.e., 5′ and 3′ UTR exon/intron variants, non-coding        patient’s phenotype terms or the causal gene.
transcript exon/intron variants, coding transcript intron variants, up-/down-              The ﬁnal DDD-derived cohort contains 1431 patients, representing
stream gene variants, intergenic variants, and regulatory region variants). We       1237 MONDO diseases and 1282 unique causal genes. Patients have 20.5
use the following pathogenicity sources, POLYPHEN, MUTATION_TASTER,                  HPO phenotype terms on average (SD = 19.2). There are 158 unique causal
and SIFT. We apply a frequency ﬁlter to remove variants with a frequency of          genes and 93 diseases found in the DDD and UDN cohorts.
at least 0.5% according to the variant frequency databases used. All variant
frequency databases are used, as recommended by the Exomiser manual. We              Simulated patients with rare Mendelian disorders
retain non-pathogenic variants in the output gene list. As with the EXPERT-          We leverage simulated but realistic rare disease patients for training
CURATED gene lists, we ﬁlter out patients who do not have at least ﬁve               SHEPHERD48. The simulated patients closely resemble real-world patients
candidate genes in their VARIANT-FILTERED gene list. The cohort includes             found in the UDN. Each simulated patient is represented by an age range, a
229 patients with at least ﬁve VARIANT-FILTERED candidate genes.                     set of positive phenotypic features they exhibit, a set of negative phenotypic
      Diagnosed patients in the UDN are labeled with a disease identiﬁer             features they do not exhibit, and a set of challenging candidate genes that
from the Online Mendelian Inheritance in Man (OMIM) database104 when                 may cause the presenting symptoms. The patients are generated using a
the patient is diagnosed with a known genetic disease. We map the OMIM               simulation framework that jointly samples candidate genes and
disease identiﬁers to MONDO identiﬁers95 using the MONDO ontology                    phenotype terms.
crosswalk to identify the diseases in the rare disease knowledge graph                     To generate patients with rare Mendelian disorders, we adopt the
(section “Rare disease knowledge graph construction”).                               pipeline described in ref. 48. Brieﬂy, the simulation pipeline has two stages:
      The ﬁnal UDN cohort contains 465 patients representing 319                     phenotype and candidate gene generation. First, each patient is initialized
MONDO diseases and 378 unique causal genes. The EXPERT-CURATED                       with a set of phenotype terms associated with a genetic disorder char-
and VARIANT-FILTERED candidate gene lists contain 244.3 and 13.3                     acterized in the rare genetic disease database Orphanet96. To reﬂect the
genes on average, respectively (SD = 244.0 and SD = 8.0). Patients have 23.9         imprecision of real-world diagnostic evaluations, the initial phenotype
HPO phenotype terms on average (SD = 16.1).                                          terms undergo phenotype dropout and corruption (i.e., phenotype terms are
                                                                                     randomly removed or replaced with more general phenotype terms), and
Patients in the MyGene2 portal                                                       additional “noisy” phenotype terms that are unrelated to the patient’s dis-
We assemble a cohort of real-world rare disease patients participating in the        ease are sampled from a large medical insurance claims database and added
MyGene2 exchange53. MyGene2, developed by the University of                          to the phenotype set. Next, candidate genes are sampled from “distractor”
Washington, is a portal through which families with rare genetic conditions          gene categories that do not cause the patient’s disease, yet would be plausible
can share their health information to connect with other families, clinicians,       candidates during the diagnostic process. The challenging distractor genes
and researchers. MyGene2 contains information about 2106 genes and the               and some of their associated phenotype terms are added. For additional
HPO phenotype terms of patients with gene mutations. MyGene2 is a                    details about the simulation process and validation of simulated patients,
member of the MatchMaker Exchange, a federated network designed to                   refer to ref. 48. To standardize across all patient cohorts, we ensure all genes
enable clinicians to ﬁnd phenotype and genotype matches for rare disease             are mapped to Ensembl identiﬁers, all diseases are mapped to MONDO
patients105. The UDN leverages the MatchMaker exchange to validate                   identiﬁers, and we construct patient subgraphs from the phenotype terms
patients’ candidate genes by ﬁnding genotype-matched individuals.                    associated with each patient.
     We retrieved data containing the sets of phenotype terms and candi-                   There are 42,624 simulated patients representing 2132 unique Men-
date genes for rare disease patients on MyGene2 as of May 7, 2022. We ﬁlter          delian disorders and 2396 unique causal genes in the simulated patient
the patients only to include patients labeled with an OMIM disease identiﬁer         dataset. Each patient is characterized by an average of 18.4 positive phe-
and a single candidate gene. This limits the cohort to patients who are likely       notype terms (SD = 7.7) and 14.0 candidate genes (SD = 3.5). Of the 378
already diagnosed. As with the other cohorts, we map all genes to Ensembl            unique causal genes and 319 unique MONDO diseases found in patients in
identiﬁers and diseases to MONDO identiﬁers, and construct patient sub-              the UDN cohort, 220 and 109 are represented in the simulated patient
graphs from the set of positive HPO terms associated with each patient.              cohort, respectively. Furthermore, 81.8% of the phenotype terms found
Demographic information, such as age, sex, or ancestry, is not systematically        across UDN patients are also found in the simulated patient cohort, and
collected in MyGene2 and is therefore unavailable for this cohort.                   29.7% of a single UDN patient’s phenotype terms are found in the most
     The ﬁnal MyGene2 cohort contains 146 patients representing 55                   similar simulated patient on average. This indicates that the simulated
MONDO diseases and 48 unique causal genes. Patients have 7.9 HPO                     patients have utility in training models that can apply to real-world UDN
phenotype terms on average (SD = 6.6). There are 14 unique causal genes              patients, but also emphasizes the need for developing models that can
and 12 diseases in the MyGene2 and UDN cohorts.                                      generalize to genes, diseases, and phenotype terms unseen at training time.

Patients derived from the Deciphering Developmental                                  Few-shot learning framework for rare disease
Disorders study                                                                      diagnosis
We construct another dataset of rare disease patients using aggregated gene          We develop SHEPHERD, a geometric deep learning approach that leverages
and phenotypic information from patients in the Deciphering Developmental            few-shot capability and external biomedical knowledge for multi-faceted

npj Digital Medicine | (2025)8:380                                                                                                                                13

## Page 14

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                   Article

diagnosis of rare diseases. SHEPHERD learns to co-embed diseases, phe-              where αv,u,k is the normalized attention weight on an edge from node v to
notypes, and genes for generating multimodal representations of rare dis-           node u computed by the k-th attention mechanism.
ease patients. As such, it performs multi-faceted diagnosis, addressing the              The third step is to update node embeddings. To transform the mes-
following challenges of rare disease diagnosis: causal gene discovery, iden-        sages into an order-invariant hidden representation hðlÞ
                                                                                                                                           v , we apply a non-
tiﬁcation of similar patients, and characterization of novel diseases.              linearity function σ and concatenate all of the aggregated messages as
     For causal gene discovery, each patient Ti in the dataset has Pi phe-          follows:
notype terms and Gi candidate genes. The task is to identify the causal
gene(s) Gci 2 Gi , harboring the variants that explain the patient’s presenting                                             K   
                                                                                                                                 ðlÞ
                                                                                                                       hðlÞ
                                                                                                                        v ¼ k σ av;k                                ð3Þ
symptoms.                                                                                                                    k¼1
     For the identiﬁcation of similar patients: Given a cohort of rare disease
patients C, the goal is to identify patients from the cohort similar to the query   In the ﬁnal layer, we perform averaging instead of concatenation. We deﬁne
patient Ti (i.e., patients that share a disease or causal gene). Mathematically,    the ﬁnal embedding for each node v after L layers of neural message passing
for each patient Ti, the task is to identify the set of patients                    as xv ¼ hvðLÞ . We specify L = 3 layers of neural message passing.
Sci ¼ fT j 2 CjGci \ Gcj ≠ ;g. We leverage each patient’s set of phenotype               Next, we deﬁne SHEPHERD’s pretraining objective function. We
terms Pi to perform patient matching.                                               frame pretraining as a binary classiﬁcation task. SHEPHERD learns to
     For the characterization of novel diseases, the goal is to characterize        perform link prediction (i.e., predict whether a relationship exists between a
novel diseases according to their similarity to a set of known genetic diseases     pair of nodes for a given relation type). Formally, we compute the score for
D. We input the set of phenotype terms Pi for each patient Ti and provide           whether an edge exists between node u and node v with relation r given their
interpretable names for the patient’s presenting syndrome.                          node embeddings xu and xv using a DistMult decoder107:

Notation                                                                                                       LPSIMðu; r; vÞ ¼ ACTðxTu Wr xv Þ                     ð4Þ
Let G denote a heterogeneous knowledge graph comprised of a set of nodes
V and a set of edges E. Each edge is deﬁned by a triplet (u, r, v) where u is the   where Wr is a relation-speciﬁc trainable weight matrix and ACT is a
source node, v is the target node, and r ∈ R denotes the relationship between       nonlinear function (here, tanh). SHEPHERD is pretrained via a hinge loss
u and v. Each patient i is represented on the graph as a patient subgraph           objective. For any pair of nodes u and v connected by relation r, the loss
induced by a set of phenotype nodes Pi where Pi ⊆ V. The patient subgraph           function is deﬁned as:
can contain any number of phenotype terms and multiple connected
components throughout G. Each patient may also have a set of candidate                          1 X
                                                                                       LLP ¼                 maxð0; Δ  LPSIMðu; r; vÞ þ LPSIMðu; r; v ÞÞ;         ð5Þ
genes Gi ⊆ V.                                                                                  jEj ðu;r;vÞ2E

SHEPHERD: encoding biomedical knowledge                                             where u and v are source and target nodes, v− is a target node representing a
The ﬁrst step in SHEPHERD is to encode biomedical relationships in the              negative example that is not linked to u in the KG, LPSIM returns the score
rare disease knowledge graph (KG). Here, we describe the architecture of            indicative of the knowledge relationship existing between u and v, and Δ is a
SHEPHERD’s GNN encoder, and then the objective function for pre-                    margin, which is set to 1 throughout all experiments in this study. For each
training SHEPHERD.                                                                  triplet (u, r, v) in the KG, its contribution to the value of the loss function is 0
      We begin by describing SHEPHERD’s GNN encoder architecture. We                if the difference between the LPSIM’s score for the triplet and the LPSIM’s
pretrain SHEPHERD on millions of biomedical entity pairs across all entity          score for a negative example is at least as large as the margin.
and relation types in the KG to capture the topology of the KG. To this end,
we use a graph attention network (GAT)106, a type of graph neural network           SHEPHERD: generating rare disease patient
(GNN) model, to generate embeddings xv for every node v in the KG.                  representations
Speciﬁcally, the choice of a graph attention network is necessary to achieve        We apply the pretrained SHEPHERD model to our multi-faceted rare
semantically-relevant mixing of biomedical entities in the embedding space;         disease diagnosis tasks. Starting with the pretrained GNN model, we learn
that is, to encourage distinct node types (e.g., genes, diseases, and pheno-        patient embeddings that encode each patient’s phenotype subgraph.
types) to be positioned near each other in the embedding space. Like most           Depending on the diagnostic task, we also learn embeddings for each
GNNs, GAT models can be formulated as message-passing networks, in                  patient’s candidate genes, diseases, or other patients. Concretely, for every
which messages are propagated to a node v from all of the nodes in its              patient Ti, we generate an aggregated representation of all phenotype terms
neighborhood N v . The messages are aggregated and combined with the                p ∈ Pi in the phenotype subgraph via a transformer encoder and an
previous layer’s representation of v to produce v’s representation for the          attention-weighted average of the individual phenotype embeddings:
current layer. Concretely, each layer l in SHEPHERD’s GNN encoder
involves the following steps:                                                                            X                                 expðxp  aÞ
      The ﬁrst step is to propagate neural messages. We deﬁne the message                       x Pi ¼          α  xp ;   where   α¼P                              ð6Þ
  ðlÞ                                                                                                    p2Pi                             p2Pi expðxp  aÞ
mv;k  for each node v as:
                                                                                    where α denotes the attention weights, xp denotes the embedding for phe-
                              mðlÞ    ðlÞ ðl1Þ
                                                                             ð1Þ    notype term p, and a is a trainable vector initialized via Xavier108. The
                               v;k ¼ Wk hv
                                                                                    aggregated phenotype representation xPi , each candidate gene node
                                                                                    embedding xg, and each candidate disease node embedding xd are pushed
where k represents the attention head, W is a trainable weight matrix, and hv
                                                                                    through two nonlinear layers to produce the embeddings zPi , zg, and zd,
is the embedding of node v in the (l − 1)-th hidden layer.
      The second step is to aggregate messages from local neighborhoods.            respectively, as:
We leverage the local neighborhood to generate a representation of each
                                                                                                         zPi     ¼ f ðf ðxPi  W1 þ b1 ÞW2 þ b2 Þ                   ð7Þ
node v. Speciﬁcally, we aggregate messages of its neighboring nodes u 2 N v
and itself using an attention mechanism to generate aðlÞv;k :
                                                                                                         zg      ¼ f ðf ðxg  W1 þ b1 ÞW2 þ b2 Þ                    ð8Þ
                                     X
                          aðlÞ
                           v;k ¼               αðlÞ      ðlÞ
                                                v;u;k  mu;k                 ð2Þ
                                   u2N v ∪ v
                                                                                                         zd      ¼ f ðf ðxd  W1 þ b1 ÞW2 þ b2 Þ                    ð9Þ

npj Digital Medicine | (2025)8:380                                                                                                                                   14

## Page 15

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                                  Article

 where f is a nonlinear function (here, leaky ReLU), and W1, W2, b1, and b2      g as follows:
 are trainable weights. The embeddings, zPi , zg, and zd, are each m-dimen-
 sional, where the output dimension m is determined via hyperparameter                             SPLSIMðP; gÞ ¼ NORMðAGGp2P ðdðp; gÞÞÞ;                                     ð13Þ
 search (section “Hyperparameter tuning”).
       Finally, each candidate gene’s embedding zg may be augmented with         where P is the patient’s phenotype terms and g is a candidate gene, AGG is
 the embeddings of the K most similar genes. First, an aggregated embedding                                                               2ðxmaxðxÞÞ
                                                                                 some aggregation function (e.g., mean), NORMðxÞ ¼ maxðxÞminðxÞ       1 is a
^zg of the K genes with the highest number of shared phenotype terms as gene
 g is generated:                                                                 normalization function to scale the values in the range [−1, 1], and d(p, g) is
                                                                                 the minimum number of hops between p and g in the KG. Whereas
             X
             K                                                                   EMBSIM captures global network topology, SPLSIM captures com-
                  simðg; hÞ
     ^zg ¼       PK              zh where simðg; hÞ ¼ jPg \ Ph j:       ð10Þ    plementary local network information via average shortest path lengths
             h¼1  i¼1 simðg; iÞ                                                  between the patient’s phenotypes and each candidate gene.
                                                                                     The ﬁnal score between a patient’s phenotype terms P and candidate
The original gene embedding is then updated via a gating mechanism as            gene g (or overall similarity) is deﬁned as:
follows:
                                                                                       SIMðP; gÞ ¼ η  EMBSIMðP; gÞ þ ð1  ηÞ  SPLSIMðP; gÞ                                   ð14Þ
                        zg ¼ ð1  θg Þ  zg þ θg  ^zg ;                 ð11Þ
                                                                                 where η is a hyperparameter ranging from [0, 1] that represents the amount
where θg controls the contribution of the original gene embedding for gene g.
                                                                                 of weight to place on EMBSIM versus SPLSIM in the ﬁnal gene prior-
We set θg ¼ ω  expðω  jN g jÞ þ 0:2 where ω is a hyperparameter for
                                                                                 itization scoring. SIM values range between [−1, 1].
the contribution of the augmented gene embedding, and jN g j is the node
                                                                                       For our objective function, we leverage a multi-similarity loss to
degree for gene g to preferentially update the embeddings for genes with
                                                                                 encourage patient phenotype embeddings to be near their causal gene
lower degree (i.e., genes that are not as well-characterized)109.
                                                                                 embedding and far away from the incorrect candidate gene embeddings.
       This approach is motivated by the observation that novel disease genes
                                                                                 The multi-similarity loss is deﬁned as follows111:
(i.e., genes without known associations with any disease) or diseases (i.e.,
diseases without known associations with any gene) can have limited prior                      0     0                                 1      0                              11
                                                                                          1X N
                                                                                               @1 log@1 þ
                                                                                                          X                               1        X
research or understanding, resulting in scarce neighbors in the knowledge          LG ¼                        expðαðSIMðPi ; gÞ  γÞÞA þ log@1 þ    expðβðSIMðPi ; gÞ  γÞÞAA;
                                                                                          N i¼1 α         g2Gc
                                                                                                                                          β         d
                                                                                                                                                         g2Gi
graph. Due to this sparsity, the gene nodes’ embeddings are of lower quality,                                    i

which can negatively impact the ability to identify the causal gene of a                                                                                                       ð15Þ
patient. For example, in an extreme case, a gene node without any con-
nections to the rest of the knowledge graph would have a randomly initi-         where N is the number of patients, α, β, and γ are hyperparameters, and
alized embedding. As such, we leverage information about shared                  SIM(Pi, g) denotes the similarity between the aggregated phenotype
phenotypic neighbors to augment these low-information gene nodes.                embedding for patient i and the gene embedding of either the patient’s
                                                                                 causal gene (g ∈ Gc) or distractor gene (g ∈ Gd). The optimized embedding
SHEPHERD: discovering causal genes                                               space encodes patient information such that the similarity between a
SHEPHERD can prioritize candidate genes to assist clinicians in ﬁnding the       patient’s phenotype terms and candidate genes (i.e., how likely it is that a
causal gene(s) harboring the variants that best explain a patient’s presenting   given gene explains the patient’s symptoms) is inversely proportional to the
symptoms. Candidate genes for each patient are scored by measuring the           distance between the patient embedding and the embedding of the
similarity SIM(P, g) between a candidate gene g and a patient’s set of phe-      candidate gene.
notype terms P. SHEPHERD is optimized such that the candidate gene
embedded near the patient’s set of phenotype terms in the embedding space        SHEPHERD: ﬁnding similar patients
indicates that the gene will likely cause the patient’s symptoms. SIM(P, g)      SHEPHERD can ﬁnd similar patients from a cohort of rare disease
consists of two components, EMBSIM(P, g) and SPLSIM(P, g). It is designed        patients. This is important for identifying molecular diagnoses and
such that EMBSIM(P, g) captures global network topology (i.e., by lever-         validating already prioritized candidate genes. To match rare disease
aging SHEPHERD’s low-dimensional embedding space) and SPLSIM(P, g)               patients, SHEPHERD ﬁrst learns a task-speciﬁc similarity function that
captures local network information (i.e., by calculating shortest path length    captures the similarity between two patients. This training process pro-
distances). This approach is grounded in the observation that, while             duces an embedding space in which the similarity between two patients is
methods that learn global network topology yield higher overall perfor-          inversely proportional to the distance between the two patient embed-
mance than local methods considering only local network information, the         dings. The embedding space can be used at inference time to answer
latter tend to rank true candidate genes higher when provided a short list of    “patients-like-me” queries. We deﬁne the similarity between two patients
candidate genes110.                                                              i and j as the L2 distance between their aggregated phenotype
      We calculate EMBSIM, an embedding-based similarity between                 embeddings zPi and zPj :
aggregated embeddings of phenotype terms P and an embedding of the
candidate gene g as follows:                                                                                    SIMðPi ; Pj Þ ¼  k zPi  zPj k22 :                            ð16Þ
                                               
                                                                                 Importantly, when calculating patient similarity, we do not include any
                     EMBSIMðP; gÞ ¼ ACT zTP Wzg                          ð12Þ
                                                                                 genotype information for the patients. This makes the model applicable in
                                                                                 settings where the patient’s genome has not been sequenced or when the
where ACT is a nonlinear function (here, tanh). EMBSIM values range              analysis results are still pending.
between [−1, 1]. Analogous to LPSIM (i.e., self-supervised link prediction),           Regarding the objective function, SHEPHERD is trained to capture
EMBSIM predicts whether there exists a relationship (i.e., harbors variants      patient similarity using the neighborhood component analysis (NCA) loss:
that explain presenting symptoms) between a patient and the                                                           P                                             !
candidate gene.                                                                                  1 X                    Pj 2Bp nPi ;Pj 2Sci      expðSIMðPi ; Pj ÞÞ
    For the network-based similarity, we calculate the shortest path length           LPH ¼                 log           P                                             ;      ð17Þ
                                                                                                 jBp j P 2B                    Pj 2Bp nPi      expðSIMðPi ; Pj ÞÞ
(SPL) similarity between aggregated phenotype terms P and candidate gene                                i   p

npj Digital Medicine | (2025)8:380                                                                                                                                              15

## Page 16

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                       Article

where Bp is a batch of patients sampled from the training set and Sci is the set          patients that are not associated with a given patient). The following outlines
of patients with the same causal gene as patient Pi. Optimizing the NCA                   the negative sampling strategies used for pretraining and each of the three
loss112 minimizes the distances between patient embeddings with the same                  rare disease diagnosis tasks.
causal gene and maximizes the distances between patient embeddings with                          For link prediction, we construct negative examples of triplets (u, r, v−)
different causal genes.                                                                   that do not exist in the KG by perturbing the target nodes while preserving
                                                                                          the types of the source and target nodes and edge relation. For example,
SHEPHERD: estimating patient-disease similarity                                           given a positive example of a triplet where the node and relation types are
Finally, SHEPHERD can characterize a clinical presentation based on                       (protein, has phenotype, phenotype), a negative example is obtained by
existing knowledge about other rare and common diseases. We analogously                   shufﬂing all phenotype nodes in the batch, thereby maintaining the node
perform novel disease characterization by learning an embedding space                     and relation types of the positive example.
such that the similarity between a patient and a disease (i.e., how likely it is                 For causal gene discovery, negative examples are constructed by taking
that a patient has that disease) is inversely proportional to the distance                the union of the candidate genes for all patients in a given batch. As noted in
between the patient embedding and the disease embedding. We deﬁne the                     sections “Patients in the Undiagnosed Diseases Network”, “Patients in the
similarity between a patient’s phenotype terms P and disease d as the L2                  MyGene2 portal”, “Patients derived from the Deciphering Developmental
distance between the aggregated phenotype embedding and the disease                       Disorders study”, and “Simulated patients with rare Mendelian disorders”,
embedding:                                                                                each patient has a list of candidate genes that have been shortlisted as the
                                                                                          most probable genes to cause the patient’s symptoms, and identifying the
                           SIMðP; dÞ ¼  k zd  zP k22                             ð18Þ   true causal gene(s) among them is especially challenging. We ensure that
                                                                                          these “hard” candidate genes are included in the candidate list for each
     Regarding the objective function to optimize patient phenotype                       patient during training, as using such “hard” examples tends to improve the
embeddings to be near their correct disease(s)', we leverage a multimodal                 efﬁciency of training113. Furthermore, to maximize the number and fre-
version of the NCA loss, deﬁned as:                                                       quency of candidate genes seen during training time, we upsample a subset
                                  P                                            !          of candidate genes that are underrepresented across all patients. Concretely,
                1 X                 d j 2Bd ;d j 2Dci   expðSIMðPi ; d j ÞÞ              we count the frequencies of candidate genes in the prior and current batches,
           LD ¼            log        P                                            ð19Þ   select the k most infrequently seen candidate genes (i.e., the k rarest can-
                jBp j P 2B                d j 2Bd expðSIMðP i ; d j ÞÞ
                       i    p                                                             didate genes) in training batches, and add them to each patient’s candidate
                                                                                          gene list. Note that we only prioritize the “hard” candidate genes for each
where Bp and Bd are batches of patients and candidate diseases, respectively,             patient at inference time without any up-sampling.
that are sampled from the training set, Pi corresponds to the phenotype term                     For novel disease characterization, negative examples include all dis-
set for patient i, and Dci is the set of correct diseases for patient i. While            eases that do not explain the patient’s symptoms. First, we randomly sample
jDci j ¼ 1 for most patients in our cohorts, several patients with multiple               1000 diseases from all diseases in the KG to serve as negative examples for
diseases exist.                                                                           each batch. Then, we calculate a patient’s similarity to all disease nodes in the
                                                                                          KG at inference time. Due to the hierarchical structure of the KG, this
Overall objective function                                                                approach may occasionally select parent or child diseases as negatives,
We train SHEPHERD in two stages. In the ﬁrst stage, we pretrain the model                 introducing potential false negatives. However, the likelihood of sampling
to learn to capture the relationships between biomedical entities in the rare             direct parent-child disease pairs as negatives is relatively low.
disease knowledge graph via self-supervised link prediction (LLP) only                           For “patients-like-me” identiﬁcation, negative examples are simply all
(section “SHEPHERD: encoding biomedical knowledge”). In the second                        of the patients in the batch who do not have the same causal gene as the
stage, we ﬁnetune the pretrained model by simultaneously predicting                       query patient. We construct batches to ensure at least two positive examples
relationships in the KG (LLP) and performing patient-centric rare disease                 (i.e., patients with the same causal gene) for each patient in the batch. All
tasks (LDX ∈ {LG, LPH, LD}) (sections “SHEPHERD: discovering causal                       remaining patients serve as negative examples. At inference time, we cal-
genes”, “SHEPHERD: ﬁnding similar patients”, and “SHEPHERD: esti-                         culate a patient’s similarity to all patients in the cohort.
mating patient-disease similarity”). In other words, the pretrained model is
jointly ﬁnetuned to achieve two distinct objectives: (1) to capture the rela-             Disease-split training on simulated and publicly avail-
tionships in the underlying knowledge graph and (2) to match a patient’s                  able patients
presenting symptoms with the patient’s causal gene(s), disease(s), or other               We train our model primarily on the simulated patient dataset. Training on
similar patients. We model these objectives with two separate loss functions,             simulated data offers several beneﬁts: the simulated cohort is larger and more
link prediction loss LLP, which continues updating node embeddings, and                   diverse than any real-world patient dataset, the trained models can be released
diagnosis loss LDX ∈ {LG, LD, LPH}, which aligns patient phenotype terms to               without the risk of exposing any patient information, and the models can be
genes, diseases, or other patient phenotypes, respectively. The overall loss              evaluated on an independent real-world cohort to test how well a model can
during the ﬁnetuning stage is as follows:                                                 generalize to patients unseen during training. Further, and most importantly,
                                                                                          we achieve generalizability to real-world cohorts by splitting patients into
                            L ¼ λLDX þ ð1  λÞLLP                                  ð20Þ   train and validation sets according to disease. Concretely, we split the list of
                                                                                          diseases represented by the simulated patient cohort into training and vali-
where λ is a hyperparameter controlling the weight of each loss. Whereas                  dation. Then, we assign patients to train or validation sets such that patients
during pretraining, we train the model to capture generalizable biomedical                with the same disease are either entirely in the training or fully in the vali-
knowledge by performing link prediction for all relation types, during ﬁne-               dation set. As a result, the model is optimized so that its parameters are
tuning, we focus on predicting gene, phenotype, and disease relations, which              broadly transferable to patients with different diseases. The resulting train and
are most important for rare diseases. Training the model to perform link                  validation cohorts contain 36,224 and 6400 patients, respectively.
prediction enables the model to generalize to phenotypes and genes unseen                      For the causal gene discovery task, we perform additional training on
in the training data.                                                                     patients from the MyGene2 and DDD cohorts. These additional cohorts
                                                                                          constitute 3.6% of the training data. Unlike the UDN cohort, the MyGene2
Negative sampling                                                                         and DDD cohorts do not have candidate genes for each patient. Therefore,
To learn a meaningful representation space, we need negative examples (i.e.,              we construct candidate gene lists by sampling 20 genes that are neighbors of
edges that do not exist in the KG, or candidate genes, diseases, or other                 each patient’s causal gene or phenotype in the rare disease knowledge graph.

npj Digital Medicine | (2025)8:380                                                                                                                                      16

## Page 17

https://doi.org/10.1038/s41746-025-01749-1                                                                                                               Article

Additional training details                                                         Performance stratiﬁed by patient and site
We provide details about node pretraining data splits, patient-driven               characteristics
sampling, and normalization.                                                        We evaluate the trained model on the cohort of real-world UDN patients
      Regarding the node pretraining data split, edges in the knowledge             who have received a molecular diagnosis (section “Patients in the Undiag-
graph are randomly split into train (80%), validation (10%), and test (10%)         nosed Diseases Network”). We measure the mean reciprocal rank of all of
sets. Note that the forward and reverse edges of the same pair of nodes are         the patients’ causal genes and calculate the percentage of causal genes that
maintained in the same data split to prevent data leakage.                          appear in the top k ranked genes for k ∈ {1, 3, 5} for the EXPERT-
      For patient-driven sampling, we design a new approach for batch               CURATED candidate gene lists and k ∈ {1, 5, 10, 25, 50} for the longer
sampling that enables the model to perform patient gene prioritization while        VARIANT-FILTERED candidate gene lists. We analyze the performance
maintaining the topology of the KG. We ﬁrst sample m patients and add               across each of the UDN clinical sites, disease categories, and evaluation
their associated phenotypes and genes to the batch. Then, we add n nodes            years. We also assess the correlation between model performance and the
randomly sampled from the genes, phenotypes, and disease nodes in the               number of patient phenotype terms, the distance between the causal gene
KG. This allows for inductive generalization by maintaining the topology of         and phenotype terms in the KG, and the prevalence of the genetic conditions
nodes not found in the training data.                                               in the population. We leverage the number of ClinVar submissions for the
      To help optimize model performance and convergence, we apply two              causal gene as a proxy for prevalence.
normalization strategies to SHEPHERD. Speciﬁcally, we use LayerNorm114
immediately after each convolutional layer and BatchNorm115 following a             Comparison to alternative approaches
nonlinear activation layer (here, leaky ReLU).                                      We compare SHEPHERD to several diverse approaches for causal gene
                                                                                    discovery. The ﬁrst category includes network science or machine learning
Hyperparameter tuning                                                               methods that enable us to assess the utility of SHEPHERD’s graph neural
We leverage Weights and Biases116 to select optimal hyperparameters via             network approach and the use of simulated patients: (1) mean shortest
a random search over the hyperparameter space. We ﬁrst choose pre-                  graph distance, a network science approach that prioritizes genes according
training hyperparameters to optimize the micro F1 score on the pretraining          to their average shortest path in the KG to all of a patient’s phenotype terms;
validation set. The pretraining validation set consists of a set of edges           (2) supervised graph embedding, a logistic regression approach that frames
that exist in the knowledge graph and a set of edges generated via                  prioritization as a binary prediction task for each candidate gene and
negative sampling that do not exist (section “Negative sampling”).                  represents each patient–gene option as the concatenation of the candidate
Hyperparameters are selected via random search from the following                   gene’s pretrained node embedding and the patient’s averaged phenotype
values: learning rate ∈ [0.0001, 0.0005, 0.001, 0.005], weight decay ∈ [0,          node embeddings; (3) supervised PCA embedding, a logistic regression
0.005, 0.0005], dropout ∈ [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], and the number of    approach similar to (2) that, instead of the KG node embeddings, utilizes a
GAT attention heads ∈ [2, 4]. We also perform a search over the                     PCA-transformed shortest path length matrix from gene nodes to gene,
dimension of the network layers: input size ∈ [2048, 4096], hidden size             phenotype, and disease nodes; and (4) large language models, namely the
∈ [256, 512, 1024], and output size ∈ [64, 128]. We then freeze the pre-            open-source LlaMa 3.1 8B and 70B parameter models55.
training hyperparameters and perform a hyperparameter search indepen-                     We also compare SHEPHERD to six phenotype-based methods
dently for each rare disease task. We select task-speciﬁc hyperparameters to        designed for causal gene discovery. There are information-theoretic
optimize the mean reciprocal rank of the correct genes, diseases, or patients       approaches that compare patient phenotype terms to either the pheno-
on the disease-split simulated validation set. Importantly, the validation          type terms associated with each candidate gene (ERIC, which is imple-
set containing simulated patients is entirely independent of the evaluation         mented in XRare25) or to the phenotype terms of all the diseases related to a
dataset, which includes patients from the Undiagnosed Diseases                      candidate gene (Phrank16 and PhenIX, which is implemented in
Network. We consider the following hyperparameters: learning rate ∈                 Exomiser24). In the latter two approaches, each candidate gene is prioritized
[0.00001, 0.00005, 0.0001, 0.0005, 0.001], λ ∈ [0.1, 0.9], η ∈ [0.1, 0.9], k most   according to the highest similarity score across all associated diseases.
infrequently seen genes ∈ [64, 128, 192], number of transformer layers              LIRICAL is an approach that uses a likelihood ratio framework to estimate
for the phenotype encoder ∈ [0, 3, 6], number of heads in the                       the extent to which patients’ phenotypes and genotypes are consistent across
transformer layers ∈ [4, 8, 16], contribution of the augmented gene                 all known diseases21. CADA is a shallow network embedding-based
embedding ω ∈ [0.1, 0.9], number of K most similar genes to augment the             approach that frames the task as link prediction between phenotype and
gene embedding ∈ [1, 2, 3, 4, 5], and number of nodes n to sample per batch         gene nodes19. HiPhive is an approach implemented in Exomiser that
in ∈ [100, 200, 300, 400]. The code for hyperparameter selection and the            leverages ontologies from humans and model organisms to assess pheno-
optimal hyperparameters can be found at https://github.com/mims-                    typic similarity. When phenotypic data is unavailable for a candidate,
harvard/SHEPHERD.                                                                   HiPhive employs a random walk approach on a protein–protein interaction
                                                                                    network to establish connections between the candidate and other genes
Implementation                                                                      with similar phenotypes24. These baselines constitute a diverse set of
We implement SHEPHERD using Pytorch (Version 1.8.0)117, Pytorch                     methodological approaches for rare disease diagnosis. LIRICAL, XRare, and
Lightning (Version 1.4.5)118, and Pytorch Geometric (Version 1.7.2)119. We          Exomiser are the strongest performing approaches in a prior comparison of
leverage the Weights and Biases116 platform for hyperparameter tuning and           gene prioritization tools121, and Exomiser is currently in use at multiple
model training visualization, and we create interactive demos of the model          UDN sites50. Finally, we compare to a non-guided baseline, Random, which
using Gradio120. Models are trained on a single NVIDIA Quadro RTX8000               represents performance without prioritizing candidate genes.
GPU. Training consists of ~15 h of pretraining and 12 h of ﬁne-tuning, with               When possible, we leverage publicly available code and our KG to
early stopping based on validation performance; these times represent upper         implement each causal gene discovery benchmark to elucidate whether per-
bounds, as the best-performing model typically emerges earlier in training.         formance differences were due to the algorithmic approach rather than a
At inference time, SHEPHERD can run on a CPU, allowing it to efﬁciently             different or more up-to-date underlying knowledge base. In instances where
rank any number of candidate genes without GPU memory constraints.                  we are unable to leverage our KG (i.e., for LIRICAL, HiPhive, and PhenIX), we
SHEPHERD processes each patient in 3.46 s on average using the EXPERT-              leverage input data that includes only the gene–phenotype–disease associa-
CURATED gene list (108 GB memory) and 3.97 s on average using the                   tions that were present at the time when our rare disease KG was constructed
VARIANT-FILTERED gene list (109 GB memory). This enables efﬁcient                   to enable a fairer comparison. We use code from https://bitbucket.org/
deployment in real-world clinical settings, including those with limited GPU        bejerano/phrank and https://github.com/Chengyao-Peng/CADA to run
resources.                                                                          Phrank and CADA, respectively, using our rare disease KG. We run Exomiser

npj Digital Medicine | (2025)8:380                                                                                                                              17

## Page 18

https://doi.org/10.1038/s41746-025-01749-1                                                                                                              Article

v13.0.1 (released Nov 23, 2021) using their time-stamped input data from            the difference in distances in the embedding space for patients with the same
https://github.com/exomiser/Exomiser/tree/13.0.1 to ensure that we only             versus different disease categories. Finally, we perform the two-sample
utilize the associations known at the time of our KG construction. We run           Kolmogorov–Smirnov test to assess whether the distribution of embedding
Exomiser with two different phenotype similarity options, PhenIX and                distances for patients with the same disease is identical to that for patients
HiPhive, and we leverage the GENE_PHENO_SCORE for prioritization. We                with different diseases.
re-implement the ERIC phenotype similarity score used in XRare; direct
gene–phenotype edges and indirect gene–disease–phenotype paths from our             Visualization of learned embeddings
KG are used to construct the phenotype terms associated with each candidate         We visualize embeddings learned via SHEPHERD in a uniform manifold
gene. We run LIRICAL with the “orphanet” data ﬂag using code from https://          approximation and projection (UMAP) plot122. We use the umap-learn
github.com/TheJacksonLaboratory/LIRICAL by supplying both a VCF ﬁle                 Python package123 and perform a grid search over the n_neighbors,
and positive and negative (i.e., that the patient did not exhibit) phenotype        min_dist, and spread UMAP parameters. We select parameters that
terms. Notably, all baselines except for LIRICAL leverage Exomiser to gen-          maintain global structure.
erate the VARIANT-FILTERED candidate genes for each patient before
phenotype-based prioritization. In contrast, LIRICAL performs its variant-          Visualization of patient neighborhoods in the
based ﬁltering using the provided VCF ﬁles.                                         knowledge graph
      We run LlaMa 3.1 8B and 70B models using the Ollama Python library            To visualize the local neighborhood of patients’ disease, phenotype, and
(https://github.com/ollama/ollama). We construct a prompt that instructs            gene nodes (Fig. 4), we calculate the shortest paths between patient-
the model to use its “knowledge of genetics, known disease–gene associa-            relevant nodes and extract all nodes in those shortest paths. We visualize
tions, and variant interpretation” to generate a ranked list of all of the          the resulting patient neighborhoods using Gephi 0.9.4124. We apply
candidate genes based on their likelihood of causing the patient’s symptoms.        Fruchterman–Reingold, Noverlap, and Label Adjust layouts and manual
The complete prompt is detailed in Supplementary Note 1. HPO terms are              adjustments to organize the nodes so they do not overlap.
included in the prompt as textual descriptions of phenotypes, and candidate
genes are provided as Ensembl IDs or Gene Symbols. We evaluate model                Data availability
performance under two conditions: when supplied with both phenotype                 All data used in the paper, including the rare disease knowledge graph,
terms and candidate genes, and when supplied only with phenotype terms.             simulated, MyGene2, and DDD patient cohorts, and the ﬁnal and inter-
If a candidate gene list is too long, the model will frequently refuse to rank      mediate results of the analyses, are shared with the research community at
the list. To address this, we split long lists containing at least 500 genes into   https://zitniklab.hms.harvard.edu/projects/SHEPHERD. The patient data-
two smaller subsets, rank each separately, and then use a prompt to merge           set derived from the Deciphering Developmental Disorders study (DDD) is
the ranked lists (see Supplementary Note 1). If the model refuses to rank the       created using aggregated gene and phenotypic information from patients in
shorter lists, we default to ranking based solely on phenotype descriptions.        the Deciphering Developmental Disorders study (https://www.
All experiments use the default hyperparameters provided by the Ollama              deciphergenomics.org/ddd/ddgenes), an initiative of nearly 14,000 pedia-
OpenAI-compatible API.                                                              tric patients with severe undiagnosed developmental disorders from the
      We also compare SHEPHERD to approaches that can be used to                    United Kingdom and Ireland. While the UDN dataset cannot be released
identify similar patients. The information-theoretic approach Phrank can be         due to privacy concerns, anonymized UDN data has been deposited in
leveraged to calculate the semantic similarity between two sets of patient          dbGaP (accession phs001232) and PhenomeCentral. Phenotypes, causal
phenotype terms based on the information content of their shared phe-               variants, and genes related to UDN diagnoses are also shared publicly in
notype ancestors in the Human Phenotype Ontology16. SET BASED cal-                  ClinVar at https://www.ncbi.nlm.nih.gov/clinvar/submitters/505999. The
culates distance between two sets of phenotype terms Pi and Pj using Jaccard        UDN study is approved by the NIH IRB Protocol 15HG0130. All patients
                                 jP \P j                                            accepted to the UDN provide written informed consent to share their data
distance, deﬁned as J ¼ 1  jP i ∪ Pj j.
                                  i    j
                                                                                    across the UDN.
     We further compare SHEPHERD to a network-based phenotype
search approach for novel disease characterization. For each patient phe-           Code availability
notype, we identify its associated diseases based on our KG (i.e., direct           Python implementation of the methodology developed and used in the
disease neighbors of each phenotype node) and retrieve the disease category         study is available via the project website at https://zitniklab.hms.harvard.
of the diseases. The percent similarity of the patient’s disease presentation to    edu/projects/SHEPHERD. The code to reproduce results, documentation,
each disease category is computed as the percentage of the associated dis-          and examples are at https://github.com/mims-harvard/SHEPHERD. We
eases in that category. These similarities become the KG-derived inter-             provide an interactive tool to explore SHEPHERD’s outputs at https://
pretable name for the patient’s novel disease presentation.                         huggingface.co/spaces/emilyalsentzer/SHEPHERD.

Assessing statistical signiﬁcance                                                   Received: 12 December 2024; Accepted: 26 May 2025;
We perform a one-sided Wilcoxon signed-rank test to assess whether there
is a signiﬁcant difference in causal gene discovery performance between
SHEPHERD and baseline methods. After conﬁrming that the data were not               References
normally distributed, we evaluate whether there is a statistically signiﬁcant       1.    Rehm, H. L. Time to make rare disease diagnosis accessible to all.
difference in SHEPHERD’s performance across clinical sites, evaluation                    Nat. Med. 28, 241–242 (2022).
years, and primary presenting symptoms using a Kruskal–Wallis H-test. In            2.    Haendel, M. et al. How many rare diseases are there? Nat. Rev. Drug
the knowledge graph, we calculate the Spearman correlation coefﬁcient to                  Discov. 19, 77–78 (2020).
measure the correlation between causal gene rank and the distance between           3.    Nguengang Wakap, S. et al. Estimating cumulative point prevalence
a patient’s phenotype terms and the causal gene. To assess whether patients               of rare diseases: analysis of the Orphanet database. Eur. J. Hum.
cluster by disease category, we perform K-means clustering with k set to the              Genet. 28, 165–173 (2020).
number of disease categories, and we evaluate the clusters according to an          4.    Whicher, D., Philbin, S. & Aronson, N. An overview of the impact of
adjusted mutual information score from scikit-learn, which is                             rare disease characteristics on research methodology. Orphanet J.
designed to consider clusters of different sizes. We assess the signiﬁcance of            Rare Dis. 13, 14 (2018).
the resulting clustering via a permutation test with 100 random permuta-            5.    Gahl, W. A. et al. The NIH undiagnosed diseases program: insights
tions of the true cluster labels. We perform a Mann–Whitney test to measure               into rare diseases. Genet. Med. 14, 51–59 (2012).

npj Digital Medicine | (2025)8:380                                                                                                                             18

## Page 19

https://doi.org/10.1038/s41746-025-01749-1                                                                                                           Article

6.    Chong, J. X. et al. The genetic basis of Mendelian phenotypes:             30.   The 100,000 Genomes Project Pilot Investigators et al. 100,000
      discoveries, challenges, and opportunities. Am. J. Hum. Genet. 97,               genomes pilot on rare-disease diagnosis in health care - preliminary
      199–215 (2015).                                                                  report. N. Engl. J. Med. 385, 1868–1880 (2021).
7.    Steinhaus, R. et al. MutationTaster2021. Nucleic Acids Res. 49,            31.   Topol, E. J. High-performance medicine: the convergence of human
      W446 (2021).                                                                     and artiﬁcial intelligence. Nat. Med. 25, 44–56. (2019).
8.    Rentzsch, P., Witten, D., Cooper, G. M., Shendure, J. & Kircher, M.        32.   Yu, K.-H., Beam, A. L. & Kohane, I. S. Artiﬁcial intelligence in
      CADD: predicting the deleteriousness of variants throughout the                  healthcare. Nat. Biomed. Eng. 2, 719–731. (2018).
      human genome. Nucleic Acids Res. 47, D886–D894 (2019).                     33.   Liu, Y. et al. A deep learning system for differential diagnosis of skin
9.    Jagadeesh, K. A. et al. M-CAP eliminates a majority of variants of               diseases. Nat. Med. 26, 900–908 (2020).
      uncertain signiﬁcance in clinical exomes at high sensitivity. Nat.         34.   Saldanha, O. L. et al. Swarm learning for decentralized artiﬁcial
      Genet. 48, 1581–1586 (2016).                                                     intelligence in cancer histopathology. Nat. Med. 28, 1232–1239
10.   Gurovich, Y. et al. Identifying facial phenotypes of genetic disorders           (2022).
      using deep learning. Nat. Med. 25, 60–64 (2019).                           35.   Bulten, W. et al. Artiﬁcial intelligence for diagnosis and Gleason
11.   Hsieh, T.-C. et al. GestaltMatcher facilitates rare disease matching             grading of prostate cancer: the PANDA challenge. Nat. Med. 28,
      using facial phenotype descriptors. Nat. Genet. 54, 349–357 (2022).              154–163 (2022).
12.   Hsieh, T.-C. et al. PEDIA: prioritization of exome data by image           36.   Ribeiro, A. H. et al. Automatic diagnosis of the 12-lead ECG using a
      analysis. Genet. Med. 21, 2807–2814 (2019).                                      deep neural network. Nat. Commun. 11, 1760 (2020).
13.   Duong, D. et al. Neural networks for classiﬁcation and image generation    37.   Tang, A. S. et al. Deep phenotyping of Alzheimer’s disease
      of aging in genetic syndromes. Front. Genet. 13, 864092 (2022).                  leveraging electronic medical records identiﬁes sex-speciﬁc clinical
14.   Hong, D. et al. Genetic syndromes screening by facial recognition                associations. Nat. Commun. 13, 675 (2022).
      technology: VGG-16 screening model construction and evaluation.            38.   Qiu, S. et al. Multimodal deep learning for Alzheimer’s disease
      Orphanet J. Rare Dis. 16, 344 (2021).                                            dementia assessment. Nat. Commun. 13, 3404 (2022).
15.   Shukla, P., Gupta, T., Saini, A., Singh, P. & Balasubramanian, R. A        39.   Tschandl, P. et al. Human-computer collaboration for skin cancer
      deep learning framework for recognizing developmental disorders.                 recognition. Nat. Med. 26, 1229–1234 (2020).
      In 2017 IEEE Winter Conference on Applications of Computer Vision          40.   De Fauw, J. et al. Clinically applicable deep learning for diagnosis
      (WACV) 705–714 (IEEE, 2017).                                                     and referral in retinal disease. Nat. Med. 24, 1342–1350 (2018).
16.   Jagadeesh, K. A. et al. Phrank measures phenotype sets similarity to       41.   Gulshan, V. et al. Development and validation of a deep learning
      greatly improve Mendelian diagnostic disease prioritization. Genet.              algorithm for detection of diabetic retinopathy in retinal fundus
      Med. 21, 464–470 (2019).                                                         photographs. JAMA 316, 2402–2410 (2016).
17.   Yang, H., Robinson, P. N. & Wang, K. Phenolyzer: phenotype-based           42.   Esteva, A. et al. Dermatologist-level classiﬁcation of skin cancer with
      prioritization of candidate genes for human diseases. Nat. Methods               deep neural networks. Nature 542, 115–118 (2017).
      12, 841–843 (2015).                                                        43.   Liang, H. et al. Evaluation and accurate diagnoses of pediatric
18.   Köhler, S. et al. Clinical diagnostics in human genetics with semantic           diseases using artiﬁcial intelligence. Nat. Med. 25, 433–438 (2019).
      similarity searches in ontologies. Am. J. Hum. Genet. 85, 457–464          44.   Boudellioua, I., Kulmanov, M., Schoﬁeld, P. N., Gkoutos, G. V. &
      (2009).                                                                          Hoehndorf, R. DeepPVP: phenotype-based prioritization of
19.   Peng, C. et al. CADA: phenotype-driven gene prioritization based on              causative variants using deep learning. BMC Bioinformatics 20, 65
      a case-enriched knowledge graph. NAR Genom. Bioinform. 3,                        (2019).
      lqab078 (2021).                                                            45.   Van Veen, D. et al. Adapted large language models can outperform
20.   Rao, A. et al. Phenotype-driven gene prioritization for rare diseases            medical experts in clinical text summarization. Nat. Med. 30,
      using graph convolution on heterogeneous networks. BMC Med.                      1134–1142 (2024).
      Genom. 11, 57 (2018).                                                      46.   Wan, P. et al. Outpatient reception via collaboration between nurses
21.   Robinson, P. N. et al. Interpretable clinical genomics with a likelihood         and a large language model: a randomized controlled trial. Nat. Med.
      ratio paradigm. Am. J. Hum. Genet. 107, 403–417 (2020).                          30, 1–8 (2024).
22.   Javed, A., Agrawal, S. & Ng, P. C. Phen-Gen: combining phenotype and       47.   Reese, J. T. et al. Systematic benchmarking demonstrates large
      genotype to analyze rare disorders. Nat. Methods 11, 935–937 (2014).             language models have not reached the diagnostic accuracy of
23.   Mao, D. et al. AI-MARRVEL - a knowledge-driven AI system for                     traditional rare-disease decision support tools. Preprint at medRxiv
      diagnosing Mendelian disorders. NEJM AI 1, AIoa2300009 (2024).                   https://doi.org/10.1101/2024.07.22.24310816 (2024).
24.   Smedley, D. et al. Next-generation diagnostics and disease-gene            48.   Alsentzer, E. et al. Simulation of undiagnosed patients with novel
      discovery with the exomiser. Nat. Protoc. 10, 2004–2015 (2015).                  genetic conditions. Nat. Commun. 14, 6403 (2023).
25.   Li, Q., Zhao, K., Bustamante, C. D., Ma, X. & Wong, W. H. Xrare: a         49.   Ramoni, R. B. et al. The undiagnosed diseases network: accelerating
      machine learning method jointly modeling phenotypes and genetic                  discovery about health and disease. Am. J. Hum. Genet. 100,
      evidence for rare disease diagnosis. Genet. Med. 21, 2126–2134                   185–192 (2017).
      (2019).                                                                    50.   Kobren, S. N. et al. Commonalities across computational workﬂows
26.   Birgmeier, J. et al. AMELIE speeds Mendelian diagnosis by matching               for uncovering explanatory variants in undiagnosed cases. Genet.
      patient phenotype and genotype to primary literature. Sci. Transl.               Med. 23, 1075–1085 (2021).
      Med. 12, eaau9113 (2020).                                                  51.   Zemojtel, T. et al. Effective diagnosis of genetic disease by
27.   Yoo, B., Birgmeier, J., Bernstein, J. A. & Bejerano, G. InpherNet                computational phenotype analysis of the disease-associated
      accelerates monogenic disease diagnosis using patients’ candidate                genome. Sci. Transl. Med. 6, 252ra123 (2014).
      genes’ neighbors. Genet. Med. 23, 1984–1992 (2021).                        52.   Chen, R. J., Lu, M. Y., Chen, T. Y., Williamson, D. F. K. & Mahmood, F.
28.   Anderson, D., Baynam, G., Blackwell, J. M. & Lassmann, T.                        Synthetic data in machine learning for medicine and healthcare. Nat.
      Personalised analytics for rare disease diagnostics. Nat. Commun.                Biomed. Eng. 5, 493–497 (2021).
      10, 5274 (2019).                                                           53.   Genomics, U. o. W. C. f. M. MyGene2. https://mygene2.org/
29.   Splinter, K. et al. Effect of genetic diagnosis on patients with                 MyGene2/.
      previously undiagnosed disease. N. Engl. J. Med. 379, 2131–2139            54.   Firth, H. V. & Wright, C. F. The deciphering developmental disorders
      (2018).                                                                          (DDD) study. Dev. Med. Child Neurol. 53, 702–703 (2011).

npj Digital Medicine | (2025)8:380                                                                                                                          19

## Page 20

https://doi.org/10.1038/s41746-025-01749-1                                                                                                         Article

55.   Dubey, A. et al. The llama 3 herd of models. Preprint at                   77.  Li, J., Cairns, B. J., Li, J. & Zhu, T. Generating synthetic mixed-type
      arXiv:2407.21783 (2024).                                                        longitudinal electronic health records for artiﬁcial intelligent
56.   Wicks, P. et al. Sharing health data for better outcomes on                     applications. npj Digit. Med. 6, 98 (2021).
      patientslikeme. J. Med. Internet Res. 12, e19. (2010).                     78. Mahmood, F. et al. Deep adversarial training for multi-organ nuclei
57.   Gerarduzzi, C. et al. Silencing SMOC2 ameliorates kidney ﬁbrosis by             segmentation in histopathology images. IEEE Trans. Med. Imag. 39,
      inhibiting ﬁbroblast to myoﬁbroblast transformation. JCI Insight 2,             3257–3267 (2020).
      e90299 (2017).                                                             79. Waheed, A. et al. CovidGAN: data augmentation using auxiliary
58.   Morkmued, S. et al. Deﬁciency of the SMOC2 matricellular protein                classiﬁer GAN for improved Covid-19 detection. IEEE Access 8,
      impairs bone healing and produces age-dependent bone loss. Sci.                 91916–91923 (2020).
      Rep.10, 14817 (2020).                                                      80. Jaipuria, N. et al. Deﬂating dataset bias using synthetic data
59.   Romio, L. et al. OFD1, the gene mutated in oral-facial-digital                  augmentation. In 2020 IEEE/CVF Conference on Computer Vision
      syndrome type 1, is expressed in the metanephros and in human                   and Pattern Recognition Workshops CVPRW (IEEE, 2020).
      embryonic renal mesenchymal cells. J. Am. Soc. Nephrol. 14,                81. Frid-Adar, M., Klang, E., Amitai, M., Goldberger, J. & Greenspan, H.
      680–689. (2003).                                                                Synthetic data augmentation using GAN for improved liver lesion
60.   Saal, S. et al. Renal insufﬁciency, a frequent complication with age in         classiﬁcation. In 2018 IEEE 15th International Symposium on
      oral-facial-digital syndrome type I. Clin. Genet. 77, 258–265 (2010).           Biomedical Imaging (ISBI) (IEEE, 2018).
61.   Ganapathi, M. et al. A homozygous splice variant in atp5po, disrupts       82. Oprisanu, B., Ganev, G. & De Cristofaro, E. On utility and privacy in
      mitochondrial complex v function and causes leigh syndrome in two               synthetic genomic data. In Proceedings of the 29th Network and
      unrelated families. J. Inherit. Metab. Dis. 45, 996–1012 (2022).                Distributed System Security Symposium (NDSS, 2022).
62.   Chen, H., Morris, M. A., Rossier, C., Blouin, J.-L. & Antonarakis, S. E.   83. Wang, Z., Myles, P. & Tucker, A. Generating and evaluating cross-
      Cloning of the cDNA for the human ATP synthase OSCP subunit                     sectional synthetic electronic healthcare data: preserving data utility
      (ATP50) by exon trapping and mapping to chromosome 21q22.                       and patient privacy. Comput. Intell. 37, 819–851 (2021).
      1-q22. 2. Genomics 28, 470–476 (1995).                                     84. Wang, J. et al. MARRVEL: integration of human and model organism
63.   Aggeler, R. et al. A functionally active human F1F0 ATPase can be               genetic resources to facilitate functional annotation of the human
      puriﬁed by immunocapture from heart tissue and ﬁbroblast cell lines:            genome. Am. Jo. Hum. Genet. 100, 843–853 (2017).
      subunit structure and activity studies. J. Biol. Chem. 277,                85. Kim, J., Wang, K., Weng, C. & Liu, C. Assessing the utility of large
      33906–33912 (2002).                                                             language models for phenotype-driven gene prioritization in the
64.   Brautigam, C. A., Wynn, R. M., Chuang, J. L. & Chuang, D. T. Subunit            diagnosis of rare genetic disease. Am. J. Hum. Genet. 111,
      and catalytic component stoichiometries of an in vitro reconstituted            2190–2202 (2024). Publisher: Elsevier.
      human pyruvate dehydrogenase complex. J. Biol. Chem. 284,                  86. Soman, K. et al. Zebra-Llama: a context-aware large language
      13086–13098 (2009).                                                             model for democratizing rare disease knowledge. Preprint at http://
65.   Jiang, Y. et al. Component co-expression and puriﬁcation of                     arxiv.org/abs/2411.02657 (2024).
      recombinant human pyruvate dehydrogenase complex from                      87. Flaharty, K. A. et al. Evaluating large language models on medical,
      baculovirus infected SF9 cells. Protein Expr. Purif. 97, 9–16 (2014).           lay-language, and self-reported descriptions of genetic conditions.
66.   Glasgow, R. I. et al. Novel GFM2 variants associated with early-                Am. J. Hum. Genet. 111, 1819–1833 (2024).
      onset neurological presentations of mitochondrial disease and              88. Chandak, P., Huang, K. & Zitnik, M. Building a knowledge graph to
      impaired expression of oxphos subunits. Neurogenetics 18,                       enable precision medicine. Sci. Data 10, 67 (2023).
      227–235 (2017).                                                            89. Marwaha, S., Knowles, J. W. & Ashley, E. A. A guide for the diagnosis
67.   Warde-Farley, D. et al. The genemania prediction server: biological             of rare and undiagnosed disease: beyond the exome. Genome Med.
      network integration for gene prioritization and predicting gene                 14, 23 (2022).
      function. Nucleic Acids Res. 38, W214–W220 (2010).                         90. Consortium, G. O. The gene ontology resource: 20 years and still
68.   Franz, M. et al. GeneMANIA update 2018. Nucleic Acids Res. 46,                  going strong. Nucleic Acids Res. 47, D330–D338 (2019).
      W60–W64 (2018).                                                            91. Jassal, B. et al. The reactome pathway knowledgebase. Nucleic
69.   Ispolatov, I., Yuryev, A., Mazo, I. & Maslov, S. Binding properties and         Acids Res. 48, D498–D503 (2020).
      evolution of homodimers in protein–protein interaction networks.           92. Piñero, J. et al. The DisGeNET knowledge platform for disease
      Nucleic Acids Res. 33, 3629–3635 (2005).                                        genomics: 2019 update. Nucleic Acids Research (2020).
70.   Keskin, O., Tuncbag, N. & Gursoy, A. Predicting protein–protein            93. Maglott, D., Ostell, J., Pruitt, K. D. & Tatusova, T. Entrez gene: gene-
      interactions from the molecular to the proteome level. Chem. Rev.               centered information at ncbi. Nucleic Acids Res. 48, D845–D855 (2005).
      116, 4884–4909 (2016).                                                     94. Köhler, S. et al. Expansion of the human phenotype ontology (HPO)
71.   Zitnik, M., Sosič, R., Feldman, M. W. & Leskovec, J. Evolution of               knowledge base and resources. Nucleic Acids Res. 47,
      resilience in protein interactomes across the tree of life. Proc. Natl          D1018–D1027 (2019).
      Acad. Sci. USA 116, 4426–4433 (2019).                                      95. Vasilevsky, N. et al. Mondo disease ontology: harmonizing disease
72.   Westermarck, J., Ivaska, J. & Corthals, G. L. Identiﬁcation of protein          concepts across the world. In CEUR-WS (2020).
      interactions involved in cellular signaling. Mol. Cell. Proteomics 12,     96. Pavan, S. et al. Clinical practice guidelines for rare diseases: the
      1752–1763 (2013).                                                               orphanet database. PLoS ONE 12, e0170365 (2017).
73.   Luck, K. et al. A reference map of the human binary protein                97. Asikainen, A., Iñiguez, G., Ureña-Carrión, J., Kaski, K. & Kivelä, M.
      interactome. Nature 580, 402–408 (2020).                                        Cumulative effects of triadic closure and homophily in social
74.   Tyler, A. L., Asselbergs, F. W., Williams, S. M. & Moore, J. H.                 networks. Sci. Adv. 6, eaax7310 (2020).
      Shadows of complexity: what biological networks reveal about               98. Kovács, I. A. et al. Network-based prediction of protein interactions.
      epistasis and pleiotropy. Bioessays 31, 220–227 (2009).                         Nat. Commun. 10, 1240 (2019).
75.   Hu, J. X., Thomas, C. E. & Brunak, S. Network biology concepts in          99. Gahl, W. A., Wise, A. L. & Ashley, E. A. The undiagnosed diseases
      complex disease comorbidities. Nat. Rev. Genet. 17, 615–629 (2016).             network of the national institutes of health: a national extension.
76.   Ried, J. S. et al. PSEA: phenotype set enrichment analysis–a new                JAMA 314, 1797–1798 (2015).
      method for analysis of multiple phenotypes. Genet. Epidemiol. 36,          100. Girdea, M. et al. Phenotips: patient phenotyping software for clinical
      244–252 (2012).                                                                 and research use. Hum. Mutat. 34, 1057–1065 (2013).

npj Digital Medicine | (2025)8:380                                                                                                                        20

## Page 21

https://doi.org/10.1038/s41746-025-01749-1                                                                                                                Article

101. Richards, S. et al. Standards and guidelines for the interpretation of         126. Menche, J. et al. Uncovering disease-disease relationships through
     sequence variants: a joint consensus recommendation of the American                 the incomplete interactome. Science 347, 1257601(2015).
     College of Medical Genetics and Genomics and the Association for               127. Oughtred, R. et al. The biogrid database: a comprehensive
     Molecular Pathology. Genet. Med. 17, 405–424 (2015).                                biomedical resource of curated protein, genetic, and chemical
102. UDN manual of operations. https://undiagnosed.hms.harvard.edu/                      interactions. Protein Sci. 30, 187–200 (2021).
     research/udn-manual-of-operations/ (2022).                                     128. Szklarczyk, D. et al. The string database in 2021: customizable
103. Robinson, P. N. et al. Improved exome prioritization of disease genes               protein–protein networks, and functional characterization of user-
     through cross-species phenotype comparison. Genome Res. 24,                         uploaded gene/measurement sets. Nucleic Acids Res. 49,
     340–348 (2014).                                                                     D605–D612 (2021).
104. Hamosh, A. et al. Online Mendelian inheritance in man (OMIM), a                129. Noori, A. Created in BioRender. https://BioRender.com/zkbpoj9
     knowledgebase of human genes and genetic disorders. Nucleic                         (2025).
     Acids Res. 33, D514–D517 (2002).                                               130. Noori, A. Created in BioRender. https://BioRender.com/26t5d3v
105. Philippakis, A. A. et al. The Matchmaker exchange: a platform for rare              (2025).
     disease gene discovery. Hum. Mutat. 36, 915–921 (2015).                        131. Noori, A. Created in BioRender. https://BioRender.com/z7vfgnl
106. Brody, S., Alon, U. & Yahav, E. How attentive are graph attention                   (2025).
     networks? ICLR (2022).
107. Yang, B., Yih, W.-t., He, X., Gao, J. & Deng, L. Embedding entities and        Acknowledgements
     relations for learning and inference in knowledge bases. ICLR (2015).          We thank Kimberly LeBlanc for reviewing our work to ensure we follow UDN
108. Glorot, X. & Bengio, Y. Understanding the difﬁculty of training deep           data privacy protocols. E.A. is supported by a Microsoft Research PhD
     feedforward neural networks. J. Mach. Learn. Res. 9, 249–256 (2010).           Fellowship. M.M.L. is supported by T32HG002295 from the National Human
109. Huang, K. et al. A foundation model for clinician-centered drug                Genome Research Institute and a National Science Foundation Graduate
     repurposing. Nat. Med. 30, 3601–3613 (2024).                                   Research Fellowship. M.Z. gratefully acknowledges the support by NSF under
110. Zolotareva, O. & Kleine, M. A survey of gene prioritization tools for          Nos. IIS-2030459 and IIS-2033384, US Air Force Contract No. FA8702-15-D-
     mendelian and complex human diseases. J. Integr. Bioinform. 16,                0001, and awards from Harvard Data Science Initiative, Amazon Research,
     20180069 (2019).                                                               Bayer Early Excellence in Science, AstraZeneca Research, and Roche Alli-
111. Wang, X., Han, X., Huang, W., Dong, D. & Scott, M. R. Multi-similarity         ance with Distinguished Scientists. UDN research reported in this manuscript
     loss with general pair weighting for deep metric learning. In 2019             was supported by the NIH Common Fund, through the Ofﬁce of Strategic
     IEEE/CVF Conference on Computer Vision and Pattern Recognition                 Coordination/Ofﬁce of the NIH Director under the following award numbers:
     (CVPR) 5017–5025 (2019).                                                       U01HG007709, U01HG010219, U01HG010230, U01HG010217,
112. Goldberger, J., Hinton, G. E., Roweis, S. & Salakhutdinov, R. R.               U01HG010233, U01HG010215, U01HG007672, U01HG007690,
     Neighbourhood components analysis. In Proc. 18th International                 U01HG007708, U01HG007703, U01HG007674, U01HG007530,
     Conference on Neural Information Processing Systems 513–520                    U01HG007942, U01HG007943, U01TR001395, U01TR002471,
     (MIT Press, 2004).                                                             U54NS108251, and U54NS093793. This study also makes use of data gen-
113. Zhao, Z.-Q., Zheng, P., Xu, S.-t. & Wu, X. Object detection with deep          erated by the DECIPHER community. A full list of centers that contributed to
     learning: a review. IEEE Trans. Neural Netw. Learn. Syst. 3212–3232            the generation of the data were available from https://deciphergenomics.org/
     (IEEE, 2019).                                                                  about/stats and via email from contact@deciphergenomics.org. DECIPHER is
114. Ba, J. L., Kiros, J. R. & Hinton, G. E. Layer normalization. NeurIPS (2016).   hosted by EMBL-EBI and funding for the DECIPHER project was provided by
115. Loffe, S. & Szegedy, C. Batch normalization: Accelerating deep                 the Wellcome Trust [grant number WT223718/Z/21/Z]. The content is solely
     network training by reducing internal covariate shift. In International        the responsibility of the authors and does not necessarily represent the ofﬁcial
     conference on machine learning, 448–456 (2015).                                views of the National Institutes of Health and other funders.
116. Biewald, L. Experiment tracking with weights and biases (2020).
117. Paszke, A. et al. Pytorch: an imperative style, high-performance               Author contributions
     deep learning library. In 33rd Conference on Neural Information                E.A., M.M.L., and S.N.K. retrieved and processed the UDN, MyGene2, DDD,
     Processing Systems (NeurIPS) (2019).                                           and simulated patient data. E.A. and M.M.L. developed, implemented, and
118. Falcon, W. & The PyTorch Lightning team. PyTorch lightning https://            benchmarked SHEPHERD and performed detailed analyses of
     github.com/PyTorchLightning/pytorch-lightning (2019).                          SHEPHERD's algorithm. E.A., M.M.L., I.S.K., and M.Z. designed the study.
119. Fey, M. & Lenssen, J. E. Fast graph representation learning with               E.A., M.M.L., S.N.K., A.N., I.S.K., and M.Z. contributed to writing the
     PyTorch Geometric. In ICLR Workshop on Representation Learning                 manuscript.
     on Graphs and Manifolds (2019).
120. Abid, A. et al. Gradio: hassle-free sharing and testing of ML models in the    Competing interests
     wild. In 2019 ICML Workshop on Human in the Loop Learning (2019).              The authors declare no competing interests.
121. Tosco-Herrera, E. et al. Evaluation of a whole-exome sequencing
     pipeline and benchmarking of causal germline variant prioritizers.             Additional information
     Hum. Mutat. 43, 2010–2020 (2022).                                              Supplementary information The online version contains
122. McInnes, L., Healy, J., Saul, N. & Großberger, L. Umap: Uniform                supplementary material available at
     manifold approximation and projection. J. Open Source Softw. 3,                https://doi.org/10.1038/s41746-025-01749-1.
     861 (2018).
123. McInnes, L. Outlier detection using UMAP – umap 0.5                            Correspondence and requests for materials should be addressed to
     documentation https://umap-learn.readthedocs.io/en/latest/                     Marinka Zitnik.
     outliers.html (2018).
124. Bastian, M., Heymann, S. & Jacomy, M. Gephi: an open source                    Reprints and permissions information is available at
     software for exploring and manipulating networks. In Proc. International       http://www.nature.com/reprints
     AAAI Conference on Web and Social Media 361–362 (2009).
125. Aken, B. L. et al. The Ensembl gene annotation system. Database                Publisher’s note Springer Nature remains neutral with regard to
     2016, baw093 (2016).                                                           jurisdictional claims in published maps and institutional afﬁliations.

npj Digital Medicine | (2025)8:380                                                                                                                               21

## Page 22

https://doi.org/10.1038/s41746-025-01749-1                                               Article

Open Access This article is licensed under a Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International License,
which permits any non-commercial use, sharing, distribution and
reproduction in any medium or format, as long as you give appropriate
credit to the original author(s) and the source, provide a link to the Creative
Commons licence, and indicate if you modiﬁed the licensed material. You
do not have permission under this licence to share adapted material
derived from this article or parts of it. The images or other third party
material in this article are included in the article’s Creative Commons
licence, unless indicated otherwise in a credit line to the material. If material
is not included in the article’s Creative Commons licence and your intended
use is not permitted by statutory regulation or exceeds the permitted use,
you will need to obtain permission directly from the copyright holder. To
view a copy of this licence, visit http://creativecommons.org/licenses/by-
nc-nd/4.0/.

© The Author(s) 2025

Undiagnosed Diseases Network

Shilpa N. Kobren1 & Isaac S. Kohane1

A full list of members and their afﬁliations appears in the Supplementary Information.

npj Digital Medicine | (2025)8:380                                                           22
