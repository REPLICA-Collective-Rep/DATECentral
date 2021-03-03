This repository documents an artistic research project that used a deep learning model to encode data produced from the moving bodies of performers for the purposes of sonification, visualization and curiosity. 

This research was conducted during rehearsals for _Dancing at the Edge of the World_, a speculative performance piece directed by Diana Neranti and supported by [The Volkswagen Foundation](https://www.volkswagenstiftung.de/).

The data were generated using a system designed by [Mika Satomi](http://www.nerding.at/cv/). More information about these suits can be found [here]().

This repository contains code to train, run and evaluate the deep neural network model used.

The repository [DatEdgeProxy](https://github.com/meredityman/DatEdgeProxy) contains code required to run an installation interactively.

---

# Introduction

## Motivation

There has long been interest in using physiological signals to drive interactive systems. The dream of intuitive, meaningful mappings between body movement and generative sound and images in a space always seems to lie just over the technological horizon.


Such data tends arrive dirty, in terms of the amount of noise, and variation between sessions, individuals and sensor placement.

We would like to extract features that represent more more human relatable properties from that data, such as tempo, tone, body position or type of movement. We hope to infer these 'high level' properties of movement from 'low level' measures of resistance in conductive fabrics.

If we attempted to handcraft features we are immediately mired, 

Its hard to sonify movement data.

In summary, data of this kind are messy.


Semantic / Edge of abstraction.

"Measurable results are not the only results"




## Research Questions

1. What gestural information from the suits can be captured, understood and modelled.
2. Can realtime sonification be used to communicate the patterns to an audience.
3. Measuring syncronicities.
4. Can we combine online and offline training to produce a system that is capable of learning adequately during the duration of a performance.


## Technical Discussion




Dimensionality reduction, or compression

[LAC : LSTM AUTOENCODER with Community for Insider Threat Detection](https://arxiv.org/pdf/2008.05646.pdf])

Online/offline

A major challenge is this project is the hight variability of the data. Small changes in sensor placement from day-to-day make it impossible to ob

Even across the course of a single days, slippage of the sensors and saturation of the fabric by sweat causes readings to drift considerably.

To mitigate this issue, a training schedule was devised that makes use of throw-away encoders for each session. Each throw-away encoder shares a single decoder that persists between sessions. We speculate that this technique should constrain the training of new encoders such that the structure of the latents space is preserved between performers and between sessions.

The cost of this approach is that encoders need to be trained afresh each session. To expedite this process a 'meta-encoder' is also trained. This meta-encoder receives updates to is weights from all of the suits. The weights of this encoder are used to initialize the throw-away encoders at the start of each session.


Supervised/Unsupervised
Online/offline



## Installation


## Data Collection

Each wearable device samples the resistivity of each of its eight sensors at 40 ms intervals. 




## Results


The 

## Further Work

- [ ] 
- [ ]



---

## Requirements


## Setup
