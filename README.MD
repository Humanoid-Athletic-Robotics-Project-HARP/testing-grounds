# Introduction

Hello! Hope all is well.

The Humanoid Athletic Robotics Project (HARP) is a team of students from [Stanford Student Robotics](https://stanfordssr.org/) with the eponymous mission of training humanoid robots to "human" like humans can... and what better test for that than athletics?

# Directories

## `runpod_docs`

Start here — this folder contains the resources needed to set up and connect to a remote machine on [runpod.io](https://runpod.io).

## `isaac_docs`

Once you've set up locally or on a RunPod, follow these instructions to configure a training environment on IsaacLab.

## `projects`

Contains running project source code. Some of these folders are submodules (separate repos or forks being actively developed).

# Setup Instructions

Clone the repo and initialize all submodules:

```bash
git clone git@github.com:Humanoid-Athletic-Robotics-Project-HARP/testing-grounds.git
cd testing-grounds
git submodule update --init --recursive
```

> **Note:** `--init --recursive` is required on a fresh clone to pull in submodule content. Without it, submodule folders will be empty.
