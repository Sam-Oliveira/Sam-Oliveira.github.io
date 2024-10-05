---
title: EczemaPF: An online learning approach to real-time eczema severity prediction.
summary: Undergraduate Thesis at the [Tanaka Group](https://www.rtanakagroup.com/) (Imperial College London).
tags:
  - Statistical ML
date: '2024-08-01'

# Optional external URL for project (replaces project detail page).
external_link: ''


links:
url_code: ''
url_pdf: ''
url_slides: ''
url_video: ''

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ''
---
Note: This work is currently being written as a paper with several other authors, and thus many details have been ommited from this page until publication.

Eczema (atopic dermatitis, or AD) is a chronic skin disease, for which designing personalised treatment strategies, as opposed to “one-size-fits-all” approaches, is of high clinical relevance. Designing personalised treatment strategies requires an accurate prediction of the evolution of eczema severity because disease symptoms fluctuate dynamically every day and manifest as relapses and remissions. Past work (EczemaPred) has used Markov-Chain Monte Carlo, an offline learning method, to predict the evolution of eczema severity scores. However, due to the offline nature of the learning algorithm, the time required for prediction grows with the number of days in the time-series data. For a time-series with about 42 days, past work takes approximately 40 seconds per patient in the dataset to output a patient’s daily prediction. This number grows without a bound as the number of days in the time-series increases, as well as the number of patients in the dataset. Thus, they are unsuitable for near real-time prediction that can guide patients towards daily treatment.

To address this problem, we propose EczemaPredFast (EczemaPF), an online inference framework that can output predictions in a matter of seconds. This framework uses Sequential Monte-Carlo to sequentially fit the model to the data as it becomes available and to predict the patient’s AD severity scores in the upcoming days. EczemaPF outperformed all reference models for 4-days-ahead PO-SCORAD prediction in both datasets after initial training. EczemaPF achieved a slightly higher performance than EczemaPred for patients with more non-zero scores that change dynamically over time, while it slightly underperformed for patients with stable and less prevalent signs. EczemaPF predicted a patient’s AD severity for the next four days in less than a second on average. In contrast, EczemaPred’s runtime grew without bounds as the size of the dataset increased; for example, runtime reached over 10 hours when the data consisted of PO-SCORAD over 77 days for 336 patients. In the future, this framework could be implemented as a severity management tool, providing patients with a prognosis of their eczema, as well as treatment recommendations to guide them towards the appropriate care required to manage their eczema severity.