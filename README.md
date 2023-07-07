This is the MovieLC dataset and code for the paper "(Knowledge Enhanced Model for Live Video Comment Generation)[https://arxiv.org/pdf/2304.14657.pdf]".
# Introduction
The “live video commenting” feature on video platforms allows users to comment at any time while watching a video, and the comments instantly slide across the video screen, making users feel like watching together with others. Automatically generating live video comments can improve user experience and enable human-like generation for bot chatting. \
Our main contribution are:
+ We build MovieLC, a movie live comments dataset. The long movie videos with informative live comments can well complement existing datasets. 
+ We propose the KLVCG model, which can generate higher-quality live video comments by effectively leveraging external knowledge. 

# Dataset
The Movie Live Comments (MovieLC) dataset contains 1,406,219 live comments from 85 movies, totaling 175 hours. Live video comments in MovieLC include divergent information associated with external knowledge, for example, the awards Sean Penn has won and other representative works in which he played the leading role, etc.
<!-- ![image error](./imgs/case.png#pic_left) -->
<div align="left">
<img src=./imgs/case.png width=60%/>
</div>

# Citation
    @article{chen2023knowledge,
    title={Knowledge Enhanced Model for Live Video Comment Generation},
    author={Chen, Jieting and Ding, Junkai and Chen, Wenping and Jin, Qin},
    journal={arXiv preprint arXiv:2304.14657},
    year={2023}
    }