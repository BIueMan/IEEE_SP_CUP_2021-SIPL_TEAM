# IEEE_SP_CUP_2021-SIPL_TEAM
The solution of SIPL_TEAM to the SP-CUP-2021 competition.

![](/images/SPCup2021.jpg)

This document describes Team SIPL’s solution for the IEEE Signal Processing Cup 2021 \cite{SPCUp2021}, in which competing teams were required to configure an Intelligent Reflecting Surface (IRS) for efficient wireless communications. To that end, we first estimated the wireless channel and then designed an algorithm that finds an IRS configuration that maximizes the rate of that channel. We began with the provided far-from-optimal IRS configurations and applied an iterative optimization technique based on gradient descent and adaptive quantization.  Further optimization is obtained by  training a deep generative neural network to find a configuration that maximizes the rate function. Each of these techniques allowed us to discover better configurations and increase data rates from the base station to the provided users. The best configurations we have discovered provide a significant improvement of the weighted average rate from 104.07 Mbit/s to 120.70 Mbit/s, compared to the best provided configurations over all users. Non IRS based solution provides average rate of 4.38~Mbit/s, which is two orders of magnitude below the IRS baseline. 

[link to the competition](https://signalprocessingsociety.org/community-involvement/ieee-signal-processing-cup-2021)

In the end we finish 6th place out of 39 competing groups. although we did not win, our solution yield better results than any other group in solving the main minimum problem, and even better then other popular solution in the field. and we were able to publishing an article on ICC 2022 conference based on our resoult. both the paper for the competition and the ICC conference are included in this Git.

This GitHub contain the code use in the compatision.

# Links To The Papers
* [SP_CUP](https://github.com/BIueMan/IEEE_SP_CUP_2021-SIPL_TEAM/blob/main/SP_Cup_2021.pdf)
* [ICC_2022](https://github.com/BIueMan/IEEE_SP_CUP_2021-SIPL_TEAM/blob/main/ICC_2021.pdf)
