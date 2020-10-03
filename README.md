# DepthEdgeAwareBENet

A. Karaali, N. Harte, CR. Jung, "Deep Multi-Scale Feature Learning for Defocus Blur Estimation", Arxiv, 2020

Any papers using this code should cite the paper accordingly. 
https://arxiv.org/abs/2009.11939

As is, the code produces the results given as the first experimental setting with dataset which is provided 
in ('Non-parametric blur map regression for depth of field extension'). 
In order to reach the dataset, you should contact with the author of this paper. 
The code here includes just 1 sample images from this dataset.

Running the code:
python3 BENet.py -i images/image_01.png

In order to make a fair comparisons, we use the same edge maps in some other edge based defocus blur 
estimations. If you want to use precomputed edges, which are provided in p"recomputed_edges/"
python3 BENet.py -i images/image_01.png -e precomputed_edges/edge_01.png 

Please also report any bug to alixkaraali[at_sign]gmail[dot_sign]com
