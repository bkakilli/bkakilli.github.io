---
title: "Yet another PyTorch Trainer/Model template"
permalink: /ya-pytorch-template/
author_profile: true
# date: "2020-08-22T00:00:00-05:00"
classes: wide
header:
  teaser: /assets/posts/pytorch-template/teaser.jpg
  # image: /assets/posts/first-flight/header.jpg
# #   overlay_color: "#000"
# #   overlay_filter: "0.5"
#   overlay_image: /assets/posts/hello_world/steven-houston-d2lO9btumD4-unsplash.jpg
#   caption: "Photo by [**Steven Houston**](https://unsplash.com/@stevenhoustonfit?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText) on [**Unsplash**](https://unsplash.com/s/photos/writing-in-the-dark?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText)"
excerpt: "A convenient project template that I build my DL models upon and train them."
# toc: true
# toc_label: "Unique Title"
# toc_icon: "heart"
---

I would like to share the template I am using for my deep learning models. I find it very convenient to start a new project on top of this. It is far from being a wrapper to PyTorch. It is just a template to design and develop DL models while conforming to the already-beautiful way of PyTorch. There are many other templates, which may provide more advanced features. [victoresque's](https://github.com/victoresque/pytorch-template){:target="_blank"} repo is a popular and great example. Although nothing stops you to use those -possibly better and more tested- alternatives, consider taking a look at mine for simplicity and ease of use.

Here is the template repo: Read below for the details.

## Index

## Project Structure
```
project_root/
    |-> models/
        |-> model1.py
        |-> model2.py
    |-> utils/
        |-> datasets/
            |-> dataset1.py
            |-> dataset2.py
        |-> data_loader.py
        |-> visualization.py
        |-> utilities1.py
        |-> utilities2.py
    |-> data/
        |-> dataset1 (can be a folder or a large dataset file)
        |-> dataset2 (can be a folder or a large dataset file)
    |-> logs/
    |-> scripts/
        |-> script1.py
        |-> script2.py
    |-> main.py
    |-> requirements.txt

    # Misc/optional files
    |-> Dockerfile
    |-> docker-compose.yaml
    |-> README.md
    |-> .gitignore
```

## Model definition
The models are defined as pure PyTorch modules. That's it. I won't go into much detail and just put a dummy model here.
```python

class FeatureExtractor(nn.Module):
    """Local feature extraction module. Uses PointNet to extract features of given point cloud.
    """
    def __init__(self, input_dim, output_dim, k=None):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(input_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):

        features = self.conv1(x.unsqueeze(-1))
        features = self.conv2(features)
        return features.squeeze(-1)


class PointClassifier(nn.Module):
    """Point classifition module for actual segmentation of the points.
    """

    def __init__(self, dims):
        super(PointClassifier, self).__init__()

        self.layers = []
        for i in range(len(dims)-2):
            self.layers += [nn.Conv2d(dims[i], dims[i+1], 1, bias=False)]
            self.layers += [nn.BatchNorm2d(dims[i+1])]
            self.layers += [nn.LeakyReLU(negative_slope=0.1)]
        self.layers += [nn.Conv2d(dims[-2], dims[-1], 1, bias=False)]

        self.classifier = nn.Sequential(*self.layers)
        
    def forward(self, X):
        return self.classifier(X)
```

## Data handling

## Training and Evaluation

## Configuration
Configuration part is where I most like it, which is featured in almost all DL projects. It allows me to experiment with many different hyperparameter settings without touching at the code itself. 