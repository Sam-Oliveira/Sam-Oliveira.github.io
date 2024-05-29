---
title: MultiDistral: Multi-Task MARL using shared distilled policies
summary: Adapting DeepMind's Distral to a Multi-Agent setting, as part of the Multi-Agent AI (COMP0124) module at UCL.
tags:
  - RL
date: '2024-04-11'

# Optional external URL for project (replaces project detail page).
external_link: ''

image:
  caption: Example MAE output.
  focal_point: Smart

links:
url_code: 'https://github.com/maxjappert/multi-agent_distral'
url_pdf: uploads/multidistral.pdf
url_slides: ''
url_video: ''

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ''
---

 Multi-task learning has been extensively applied to reinforcement learning problems to counteract its data inefficiency. One possible approach is based on transferring knowledge via policies: Due to the similarity between tasks, an agent’s policy for one task is likely to share similarities with its policy for other tasks. This idea has been applied to single-agent RL to create the Distral Framework: An agent aims to learn a policy for each task, while constrained to find a policy that is similar to a shared policy. 
 
 Based on this idea, we propose MultiDistral, an extension of Distral to the multi-agent setting. We show that  MultiDistral outperforms a Q-Learning baseline given few games played. However, we observe that learning with MultiDistral in a semi-collaborative setting results in one player's performance worsening as more iterations are run. We hypothesise from behaviour simulations that this is due to competitiveness, as the better player learns to dominate the competitive resource, resulting in the disadvantaged player having to spend many time steps without the ability to access it and thus learn. We finish by studying the differences between two versions of this framework, as well as by analysing the impact of the different tasks' characteristics on the learning process. 