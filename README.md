# Multimodal_Style_Transfer
Image Style Transfer with style comes from text description(CPU friendly).

## Setup

#### You are recommended to create a virtual envirnment with conda

```shell script
conda create -n StyleTransfer python=3.10

conda activate StyleTransfer
```

#### Then download the necessary Python packages through

```shell script
pip install -r requirement.txt
```

## Run

#### Run StyleTransfer with

```shell script
python transfer.py
```

And here are examples for what it would look like if you use `Fire` as text description.

Content          |  Target
:-------------------------:|:-------------------------:
![](./data/face2.jpeg)  |  ![](./outputs/Fire_face2_exp1.jpg)

Hopefully you will like it.