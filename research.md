# Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification



## ABSTRACT

Recently, self-attention mechanisms have shown impressive performance in various NLP and CV tasks, which can help capture sequential characteristics and derive global information. In this work, we explore how to extend self-attention modules to better learn subtle feature embeddings for recognizing fine-grained objects, e.g., different bird species or person identities. To this end, we propose a dual cross-attention learning (DCAL) algorithm to coordinate with self-attention learning. First, we propose global-local cross-attention (GLCA) to enhance the interactions between global images and local high-response regions, which can help reinforce the spatial-wise discriminative clues for recognition. Second, we propose pair-wise cross-attention (PWCA) to establish the interactions between image pairs. PWCA can regularize the attention learning of an image by treating another image as distractor and will be removed during inference. We observe that DCAL can reduce misleading attentions and diffuse the attention response to discover more complementary parts for recognition. We conduct extensive evaluations on fine-grained visual categorization and object re-identification. Experiments demonstrate that DCAL performs on par with state-of-the-art methods and consistently improves multiple self-attention baselines, e.g., surpassing DeiT-Tiny and ViT-Base by 2.8\% and 2.4\% mAP on MSMT17, respectively.

## Introduction

Self-attention is an attention mechanism that can relate different positions of a single sequence and draw global dependencies. It is originally applied in natural language processing (NLP) tasks \cite{vaswani2017attention,devlin2018bert} and exhibits the outstanding performance. Recently, Transformer with self-attention learning has also been explored for various vision tasks (e.g., image classification \cite{dosovitskiy2020image,chen2020generative,touvron2020training,ramachandran2019stand,hu2019lrnet,wang2020axial} and object detection \cite{carion2020detr,zhu2020deformable}) as an alternative of convolutional neural network (CNN). For general image classification, self-attention has been proved to work well for recognizing 2D images by viewing image patches as words and flattening them as sequences \cite{dosovitskiy2020image,touvron2020training}.

In this work, we investigate how to extend self-attention modules to better learn subtle feature embeddings for recognizing fine-grained objects, e.g., different bird species or person identities. Fine-grained recognition is more challenging than general image classification owing to the subtle visual variations among different sub-classes. Most of existing approaches build upon CNN to predict class probabilities or measure feature distances. To address the subtle appearance variations, local characteristics are often captured by learning spatial attention \cite{fu2017look,zheng2017learning,sun2018multi,luo2019cross} or explicitly localizing semantic objects / parts \cite{zheng2019looking,ding2019selective,yang2018learning, zhang2019learning}.
%dong: add reid references
We adopt a different way to incorporate local information based on vision Transformer. To this end, we propose global-local cross-attention (GLCA) to enhance the interactions between global images and local high-response regions. Specifically, we compute the cross-attention between a selected subset of query vectors and the entire set of key-value vectors. By coordinating with self-attention learning, GLCA can help reinforce the spatial-wise discriminative clues to recognize fine-grained objects.

Apart from incorporating local information, another solution to distinguish the sutble visual differences is pair-wise learning. The intuition is that one can identify the subtle variations by comparing image pairs. Exiting CNN-based methods design dedicated network architectures to enable pair-wise feature interaction \cite{zhuang2020learning,gao2020channel}. A contrastive loss \cite{gao2020channel} or score ranking loss \cite{zhuang2020learning} is used for feature learning. Motivated by this, we also employ a pair-wise learning scheme to establish the interactions between image pairs. Different from optimizing the feature distance, we propose pair-wise cross-attention (PWCA) to regularize the attention learning of an image by treating another image as distractor. Specifically, we compute the cross-attention between query of an image and combined key-value from both images. By introducing confusion in key and value vectors, the attention scores are diffused to another image so that the difficulty of the attention learning of the current image increases. Such regularization allows the network to discover more discriminative regions and alleviate overfitting to sample-specific features. It is noted that PWCA is only used for training and thus does not introduce extra computation cost during inference. 

The proposed two types of cross-attention are easy-to-implement and compatible with self-attention learning. We conduct extensive evaluations on both fine-grained visual categorization (FGVC) and object re-identification (Re-ID). Experiments demonstrate that DCAL performs on par with state-of-the-art methods and consistently improves multiple self-attention baselines. Particularly, for FGVC, DCAL improves DeiT-Tiny by 2.5\% and reaches 92.0\% top-1 accuracy with the larger R50-ViT-Base backbone on CUB-200-2011. For Re-ID, DCAL improves DeiT-Tiny and ViT-Base by 2.8\% and 2.4\% mAP on MSMT17, respectively.

Our main contributions can be summarized as follows. (1) We propose global-local cross-attention to enhance the interactions between global images and local high-response regions for reinforcing the spatial-wise discriminative clues. (2) We propose pair-wise cross-attention to establish the interactions between image pairs by regularizing the attention learning. (3) The proposed dual cross-attention learning can complement the self-attention learning and achieves consistent performance improvements over multiple vision Transformer baselines on various FGVC and Re-ID benchmarks. 


## Proposed Approach


### Revisit Self-Attention

\cite{vaswani2017attention} originally proposes the self-attention mechanism to address NLP tasks by calculating the correlation between each word and all the other words in the sentence. \cite{dosovitskiy2020image} inherits the idea by taking each patch in the image / feature map as a word for general image classification. In general, a self-attention function can be depicted as mapping a query vector and a set of key and value vectors to an output. The output is computed as a weighted sum of value vectors, where the weight assigned to each value is computed by a scaled inner product of the query with the corresponding key. Specifically, a query $q \in \mathbb{R}^{1\times d}$ is first matched against $N$ key vectors ($K=[k_1;k_2;\cdots ;k_N]$, where each $k_i \in \mathbb{R}^{1\times d}$) using inner product. The products are then scaled and normalized by a softmax function to obtain $N$ attention weights. The final output is the weighted sum of $N$ value vectors ($V=[v_1;v_2;\cdots ;v_N]$, where each $v_i \in \mathbb{R}^{1\times d}$). By packing $N$ query vector into a matrix $Q=[q_1;q_2;\cdots ;q_N]$, the output matrix of self-attention (SA) can be represented as:
$$

f_{\text{SA}}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V = SV

$$
where $\frac{1}{\sqrt{d}}$ is a scaling factor.  Query, key and value matrices are computed from the same input embedding $X \in \mathbb{R}^{N\times D}$ with different linear transformations: $Q=XW_Q$, $K=XW_K$, $V = XW_V$, respectively. $S \in \mathbb{R}^{N\times N}$ denotes the attention weight matrix. 

To jointly attend to information from different representation subspaces at different positions, multi-head self-attention (MSA) is defined by considering multiple attention heads. The process of MSA can be computed as linear transformation on the concatenations of self-attention blocks with subembeddings. To encode positional information, fixed / learnable position embeddings are added to patch embeddings and then fed to the network. To predict the class, an extra class embedding $\hat{\texttt{CLS}} \in \mathbb{R}^{1\times d}$ is prepended to the input embedding $X$ throughout the network, and finally projected with a linear classifer layer for prediction. Thus, the input embeddings as well as query, key and value matrices become $(N+1)\times d$ and the self-attention function (Eq. \ref{eq:sa}) allows to spread information between patch and class embeddings. 
%The class embedding is learnable and the training loss is imposed on the final output of classifer layer. 

Based on self-attention, a Transformer encoder block can be constructed by an MSA layer and a feed forward network (FFN). FFN consists of two linear transformation with a GELU activation. Layer normalization (LN) is put prior to each MSA and FFN layer and residual connections are used for both layers.


### Global-Local Cross-Attention
Self-attention treats each query equally to compute global attention scores according to Eq. \ref{eq:sa}. In other words, each local position of image is interacted with all the positions in the same manner. For recognizing fine-grained objects, we expect to mine discriminative local information to facilitate the learning of subtle features. To this end, we propose global-local cross-attention to emphasize the interaction between global images and local high-response regions. First, we follow attention rollout \cite{abnar2020quantifying} to calculate the accumulated attention scores for $i$-th block:

$$


\hat{S}_i = \bar{S}_i \otimes \bar{S}_{i-1} \cdots \otimes \bar{S}_1
$$

where $\bar{S}=0.5S+0.5E$ means the re-normalized attention weights using an identity matrix $E$ to consider residual
connections, $\otimes$ means the matrix multiplication operation. In this way, we track down the information propagated
from the input layer to a higher layer. Then, we use the aggregated attention map to mine the high-response regions. According to Eq. \ref{eq:rollout}, the first row of $\hat{S}_i = [\hat{s}_{i,j}]_{(N+1)\times (N+1)}$ means the accumulated weights of class embedding $\hat{\texttt{CLS}}$. We select top $R$ query vectors from $Q_i$ that correspond to the top $R$ highest responses in the accumulated weights of $\hat{\texttt{CLS}}$ to construct a new query matrix $Q^l$, representing the most attentive local embeddings. Finally, we compute the cross attention between the selected local query and the global set of key-value pairs as below.

$$
f_{\text{GLCA}}(Q^l,K^g,V^g)=\text{softmax}(\frac{Q^l{K^g}^T}{\sqrt{d}})V^g

$$

In self-attention (Eq. \ref{eq:sa}), all the query vectors will be interacted with the key-value vectors. In our GLCA (Eq. \ref{eq:glca}), only a subset of query vectors will be interacted with the key-value vectors. We observe that GLCA can help reinforce the spatial-wise discriminative clues to promote recognition of fine-grained classes. Another possible choice is to compute the self-attention between local query $Q^l$ and local key-value vectors ($K^l$, $V^l$). However, through establishing the interaction between local query and global key-value vectors, we can relate the high-response regions with not only themselves but also with other context outside of them. Figure \ref{figure:overview} (a) illustrates the proposed global-local cross-attention and we use $M=1$ GLCA block in our method. 


### Pair-Wise Cross-Attention
The scale of fine-grained recognition datasets is usually not as large as that of general image classification, e.g., ImageNet \cite{deng2009imagenet} contains over 1 million images of 1,000 classes while CUB \cite{wah2011caltech} contains only 5,994 images of 200 classes for training. Moreover, smaller visual differences between classes exist in FGVC and Re-ID compared to large-scale classification tasks. Fewer samples per class may lead to network overfitting to sample-specific features for distinguishing visually confusing classes in order to minimize the training error. 

To alleviate the problem, we propose pair-wise cross attention to establish the interactions between image pairs. PWCA can be viewed as a novel regularization method to regularize the attention learning. Specifically, we randomly sample two images ($I_1$, $I_2$) from the same training set to construct the pair. The query, key and value vectors are separately computed for both images of a pair. For training $I_1$, we concatenate the key and value matrices of both images, and then compute the attention between the query of the target image and the combined key-value pairs as follows:
$$

f_{\text{PWCA}}(Q_1,K_c,V_c) =\text{softmax}(\frac{Q_1 K_c^T}{\sqrt{d}})V_c
$$

where $K_c=[K_1;K_2] \in \mathbb{R}^{(2N+2)\times d}$ and $V_c=[V_1;V_2] \in \mathbb{R}^{(2N+2)\times d}$. For a specific query from $I_1$, we compute $N+1$ self-attention scores within itself and $N+1$ cross-attention scores with $I_2$ according to Eq. \ref{eq:pwca}. All the $2N+2$ attention scores are normalized by the softmax function together and thereby contaminated attention scores for the target image $I_1$ are learned. 

Optimizing this noisy attention output increases the difficulty of network training and reduces the overfitting to sample-specific features. Figure \ref{figure:overview} (b) illustrates the proposed pair-wise cross-attention and we use $T=12$ PWCA blocks in our method. Note that PWCA is only used for training and will be removed for inference without consuming extra computation cost.  


## Experiments




### Experimental Setting
**Datasets.**
We conduct extensive experiments on two fine-grained recognition tasks: fine-grained visual categorization (FGVC) and object re-identification (Re-ID). For FGVC, we use three standard benchmarks for evaluations: CUB-200-2011 \cite{wah2011caltech}, Stanford Cars \cite{krause20133d}, FGVC-Aircraft \cite{maji2013fine}.
%and Stanford Dogs \cite{khosla2011novel}. 
For Re-ID, we use four standard benchmarks: Market1501 \cite{zheng2015scalable}, DukeMTMC-ReID \cite{wu2020deep}, MSMT17 \cite{wei2018person} for Person Re-ID and VeRi-776 \cite{zheng2020vehiclenet} for Vehicle Re-ID. In all experiments, we use the official train and validation splits for evaluation.

**Baselines.**
We use DeiT and ViT as our self-attention baselines. In detail, ViT backbones are pre-trained on ImageNet-21k \cite{deng2009imagenet} and DeiT backbones are pre-trained on ImageNet-1k \cite{deng2009imagenet}. We use multiple architectures of DeiT-T/16, DeiT-S/16, DeiT-B/16, ViT-B/16, R50-ViT-B/16 with $L=12$ SA blocks for evaluation.

**Implementation Details.**
We coordinate the proposed two types of cross-attention with self-attention in the form of multi-task learning. We build $L=12$ SA blocks, $M=1$ GLCA blocks and $T=12$ PWCA blocks as the overall architecture for training. The PWCA branch shares weights with the SA branch while GLCA does not share weights with SA. We follow \cite{zhang2021fairmot} to adopt dynamic loss weights for collaborative optimization, avoiding exhausting manual hyper-parameter search. The PWCA branch has the same GT target as the SA branch since we treat another image as distractor.

For FGVC, we resize the original image into 550$\times$550 and randomly crop to 448$\times$448 for training. The sequence length of input embeddings for self-attention baseline is $28\times 28=784$. We select input embeddings with top $R=10\%$ highest attention responses as local queries. We apply stochastic depth \cite{huang2016deep} and use Adam optimizer with weight decay of 0.05 for training. The learning rate is initialized as ${\rm lr}_{scaled}=\frac{5e-4}{512}\times batchsize$ and decayed with a cosine policy. We train the network for 100 epochs with batch size of 16 using the standard cross-entropy loss. 

For Re-ID, we resize the image into 256$\times$128 for pedestrian datasets, and 256$\times$256 for vehicle datasets. We select input embeddings with top $R=30\%$ highest attention responses as local queries. We use SGD optimizer with a momentum of 0.9 and a weight decay of 1e-4. The batch size is set to 64 with 4 images per ID. The learning rate is initialized as 0.008 and decayed with a cosine policy. We train the network for 120 epochs using the cross-entropy and triplet losses.

All of our experiments are conducted on PyTorch with Nvidia Tesla V100 GPUs. Our method costs 3.8 hours with DeiT-Tiny backbone for training using 4 GPUs on CUB, and 9.5 hours with ViT-Base for training using 1 GPU on MSMT17. During inference, we remove all the PWCA modules and only use the SA and GLCA modules. We add class probabilities output by classifiers of SA and GLCA for prediction for FGVC, and concat two final class tokens of SA and GLCA for prediction for Re-ID. A single image with the same input size as training is used for test. 


%-------------------------------------------------------------------------




### Results on Fine-Grained Visual Categorization

We evaluate our method on three standard FGVC benchmarks and compare with the state-of-the-art approaches in Table \ref{fine-grained sota compare}. Our method achieves competitive performance compared to the prior CNN-based and Transformer-based methods. Particularly, with the R50-ViT-Base backbone, DCAL reaches 92.0\%, 95.3\% and 93.3\% top-1 accuracy on CUB-200-2011, Stanford Cars and FGVC-Aircraft benchmarks, respectively. Table \ref{fine-grained sota compare} also shows our method can consistently improve different vision Transformer baselines on all the three benchmarks, e.g., surpassing the pure Transformer (DeiT-Tiny) by 2.2\% and the hybrid structure of CNN and Transformer (R50-ViT-Base) by 1.3\% on Stanford Cars. The results validate the compatibility of our method to different Transformer architectures. 

**Comparisons to Transformer-based Methods.**
Our method performs on par with the recent Transformer variants on FGVC: TransFG \cite{he2021transfg}, RAMS-Trans \cite{hu2021rams}, FFVT \cite{wang2021feature}. These existing methods also select tokens based on aggregated attention responses. Differently, they continue to model the selected tokens by self-attention while we perform cross-attention between local query and global key-value vectors. Compared to self-attention in selected tokens, we can relate the high-response regions with not only themselves but also with other context outside of them. Besides, TransFG \cite{he2021transfg} uses overlapping patches and will largely increase training time and computation overhead, while we adopt the standard non-overlapping patch split method.


**Comparisons to CNN-based Methods.**
(1) Existing region-based methods can be divided to two categories. Explicit localization methods (e.g, RACNN \cite{fu2017look}, MA-CNN \cite{zheng2017learning}, NTS-Net \cite{yang2018learning}, MGE-CNN \cite{zhang2019learning}) utilize attention / localization sub-network with ranking losses to mine object regions. Implicit localization methods (e.g., S3N \cite{ding2019selective}, TASN \cite{zheng2019looking}) use class activation map and Gaussian sampling to amplify object regions in the original image. Our GLCA adopts a different scheme to incorporate the local information with higher performance, e.g., +3.5\% over MGE-CNN on CUB. (2) Pair-wise learning is also applied for FGVC by interacting features (CIN \cite{gao2020channel}, API-Net \cite{zhuang2020learning}) or introducing confusion (PC \cite{dubey2018pairwise}, SPS \cite{huang2021stochastic}) between image pairs during training. Our motivation of PWCA is similar to \cite{dubey2018pairwise,huang2021stochastic} but we implement a different regularization method to alleviate overfitting. Our method surpasses these related pair-wise learning methods, e.g., +3.9\% over CIN and +5.1\% over PC on CUB.


### Results on Object Re-ID

We evaluate our method on four standard Re-ID benchmarks in Table \ref{reid sota compare} and achieve competitive performance compared to the state-of-the-art methods on both Person Re-ID and Vehicle Re-ID tasks. Particularly, with the ViT-Base backbone, DCAL reaches 80.2\%, 64.0\%, 87.5\%, 80.1\% mAP on VeRi-776, MSMT17, Market1501, DukeMTMC, respectively. Similar to FGVC, our method can consistently improve different vision Transformer baselines, e.g., surpassing the light-weight Transformer (DeiT-Tiny) by 2.8\% and the larger Transformer (ViT-Base) by 2.4\% on MSMT17. 


**Comparisons to Transformer-based Methods.**
Our method performs on par with the recent Transformer variants on Re-ID: DRL-Net \cite{jia2021drl}, AAformer \cite{zhu2021aaformer}, TransReID \cite{he2021transreid}. DRL-Net \cite{jia2021drl} imposes decorrelation constraints on Transformer decoder to disentangle ID relevant and irrelevant features, while we only employ Transformer encoder and extend self-attention to cross-attention. Both of existing methods (TransReID \cite{he2021transreid}, AAformer \cite{zhu2021aaformer}) and our methods incorporate local information for recognition but adopt different manners. TransReID \cite{he2021transreid} designs a jigsaw patch module to shuffle the patch embeddings for learning robust features. AAformer \cite{zhu2021aaformer} computes the attention between a part token and its associated subset of patch embeddings by online clustering. Differently, we proposes global-local cross-attention to enhance the interactions between global images and local regions. 

**Comparisons to CNN-based Methods.** (1) Many prior approaches have been presented to encode discriminative part-level features for recognition. Typical part-based ReID methods include SPReID \cite{kalayeh2018spreid} and PCB \cite{sun2018pcb}. SPReID \cite{kalayeh2018spreid} utilizes a parsing model to generate human part masks to compute reliable part representations, which consumes extra computation overhead in segmentation part. PCB \cite{sun2018pcb} utilizes a refined part pooling to retrieve the body part information. Our method does not aim to mine precise object parts but establish the interactions between global images and high-response local regions. (2) Image pairs or triplets are widely used in Re-ID for metric learning. Recent Re-ID methods also introduce pair-wise spatial transformer to match the holistic and partial image pairs \cite{luo2020stnreid} or design pair-wise loss to learn fine-grained features for recognition \cite{yan2021beyond}. Our pair-wise cross-attention is a new practice in Re-ID in contrast to previous work.


### Ablation Study

**Contributions from Algorithmic Components.** We examine the contributions from the two types of cross-attention modules using different vision Transformer baselines in Table \ref{table:glca_pwca_module}. We use DeiT-Tiny for FGVC and ViT-Base for Re-ID. With either GLCA or PWCA alone, our method can obtain higher performance than the baselines. With both cross-attention modules, we can further improve the results. We note that PWCA will be removed for inference so that it does not introduce extra parameters or FLOPs. We uses one GLCA module in our method, which only requires a small increase of parameters or FLOPs compared to the baseline.

**Ablation Study on GLCA.** (1) Cross-ViT \cite{chen2021crossvit} is a most recent method based on cross-attention for general image classification. It constructs two Transformer branches to handle image tokens of different sizes and uses the class token from one branch to interact with patch tokens from another branch. We implement this idea using the same selected local queries and the same DeiT-Tiny backbone. The cross-token strategy obtains 82.1\% accuracy on CUB, which is worse than our GLCA by 1\%. (2) Another possible baseline to incorporate local information is computing the self-attention for the high-response local regions (i.e., local query, key and value vectors). This local self-attention baseline obtains 82.6\% accuracy on CUB using the DeiT-Tiny backbone, which is also worse than our GLCA (83.1\%). (3) We conduct more ablation experiments to examine the effect of GLCA. We obtain 82.6\% accuracy on CUB by selecting local query randomly and obtain 82.8\% by selecting local query based on the penultimate layer only. Our GLCA outperforms both baselines, validating that mining high-response local query with aggregated attention map is effective for our cross-attention learning.

**Caption:** Comparisons of different regularization methods. DeiT-Tiny is used for CUB and ViT-Base is used for MSMT17.


**Ablation Study on PWCA.**
We compare PWCA with different regularization strategies in Table \ref{regularization methods.} by taking $I_1$ as the target image. The results show that adding image noise or label noise without cross-attention causes degraded performance compared to the self-attention learning baseline. As the extra image $I_2$ used in PWCA can be viewed as distractor, we also test replacing the key and value embeddings of $I_2$ with Gaussian noise. Such method performs better than adding image / label noise, but still worse than our method. Moreover, sampling $I_2$ from a different dataset (i.e., COCO), sampling intra-class / inter-class pair only, or sampling intra-class \& inter-class pairs with equal probability performs worse than PWCA. We assume that the randomly sampled image pairs from the same dataset (i.e., natural distribution of the dataset) can regularize our cross-attention learning well.


**Amount of Cross-Attention Blocks.** Figure \ref{figure:block} presents the ablation experiments on the amount of our cross-attention blocks using DeiT-Tiny for CUB and ViT-Base for MSMT17. For GLCA, the results show that $M=1$ performs best. We analyze that the deeper Transformer encoder can produce more accurate accumulated attention scores as the attention flow is propagated from the input layer to higher layer. Moreover, using one GLCA block only introduces small extra Parameters and FLOPs for inference. For PWCA, the results show that $T=12$ performs best. It implies that adding $I_2$ throughout all the encoders can sufficiently regularize the network as our self-attention baseline has $L=12$ blocks in total. Note that PWCA is only used for training and will be removed for inference without consuming extra computation cost.


### Limitations
Compared to the self-attention learning baseline, our method may take longer time for network convergence as we perform joint training of self-attention and the proposed two types of cross-attention. For example, the self-attention baseline costs 2.1 hours while our method costs 3.8 hours for training on CUB with the same DeiT-backbone and same epochs of 100. However, it is noted that fine-grained recognition datasets are much smaller than the large-scale image classification benchmark and thereby our training time in practice is still acceptable.

Another limitation is that GLCA will increase small computation cost compared to the self-attention baseline. For example, Table \ref{table:glca_pwca_module} shows that GLCA increases 9\% Params and 2\% FLOPs for DeiT-Tiny on CUB and increases 8\% Params and 3\% FLOPs for ViT-Base on VeRi-776. We also test removing both GLCA and PWCA blocks for maintaining the same computation cost with the self-attention baseline, and the performance slightly drops, e.g, 84.3\% vs. 84.6\% (Ours) accuracy on CUB and 80.1\% vs. 80.2\% (Ours) mAP on VeRi-776.


%------------------------------------------------------------------------
## {Conclusion}
In this work, we introduce two types of cross-attention mechanisms to better learn subtle feature embeddings for recognizing fine-grained objects. GLCA can help reinforce the spatial-wise discriminative clues by modeling the interactions between global images and local regions. PWCA can establish the interactions between image pairs and can be viewed as a regularization strategy to alleviate overfitting. Our cross-attention design is easy-to-implement and compatible to different vision Transformer baselines. Extensive experiments on seven benchmarks have demonstrated the effectiveness of our method on FGVC and Re-ID tasks. We expect that our method can inspire new insights for the self-attention learning regime in Transformer.

\clearpage
\appendix



## Overview
In this supplementary material, we present more experimental results and analysis.
\begin{itemize}
    \item We test different inference architectures. 
    \item We provide additional ablation study on effect of ratio of local query selection.
    \item We show more visualization results of generated attention maps on different benchmarks. 
    \item We conduct experiments on more Transformer baselines.
\end{itemize}

## Different Inference Architectures
Our default inference architecture is that all the PWCA modules are removed and only SA and GLCA modules are used. For FGVC, we add class probabilities output by classifiers of SA and GLCA for prediction. For Re-ID, we concat two final class tokens of SA and GLCA as the output feature for prediction. We also test two different inference architectures: (1) ``SA``: using the last SA module for inference. (2) ``GLCA``: using the GLCA module for inference. Table \ref{table:fine-grained} and \ref{table:reid} present the detailed performance with different baselines on all the FGVC and Re-ID benchmarks, respectively. The results show that only using the SA or GLCA module can obtain similar performance with our default setting. It is also noted that ``SA`` has the same inference architecture with the baseline by removing all the PWCA and GLCA modules for inference, which does not introduce extra computation cost.


## Ablation Study on Effect of $R$
We test different choices of the ratios of selecting high-response regions as local query. Figure \ref{figure:r} shows that different choices of $R$ can obtain similar performance. We set $R=10\%$ for all the FGVC benchmarks and set $R=30\%$ for all the Re-ID benchmarks as default in our method. 

## More Visualization Results
We show more visualization results by comparing self-attention and our cross-attention method. Figure \ref{figure: heatmap_cub}, \ref{figure: heatmap_car}, \ref{figure: heatmap_air} present the generated attention maps on different FGVC benchmarks. Figure \ref{figure: heatmap_market}, \ref{figure: heatmap_duke}, \ref{figure: heatmap_veri} present the generated attention maps on different Re-ID benchmarks. The results show that our DCAL can reduce misleading attentions and diffuse the attention response to discover more complementary parts for recognition.

## More Transformer Baselines
We conduct two more experiments on CaiT \cite{touvron2021going} and Swin Transformer \cite{liu2021swin}. CaiT-XS24 obtains 88.5\% while our method obtains 89.7\% top-1 accuracy on CUB. Swin-T obtains 84.9\% while our method obtains 85.8\% top-1 accuracy on CUB. For Re-ID on MSMT, Swin-T achieves 55.7\% while we achieve 56.7\% mAP. As locality has been incorporated by windows in Swin Transformer, we only apply PWCA into it. 