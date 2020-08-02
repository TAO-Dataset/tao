# TAO ECCV'20 Multi-Object Tracking Challenge

We are excited to host a challenge on TAO as part of our
[ECCV workshop](http://taodataset.org/workshop/).
The challenge is hosted on the [motchallenge.net](https://motchallenge.net/) website:
[link](https://motchallenge.net/results/ECCV_2020_TAO_Challenge/).

## Important Dates

- July 10: Challenge released!
- August 16: Challenge closes, winners contacted to prepare presentation for ECCV workshop.
- August 23: ECCV workshop date. Challenge results announced, along with
presentations by challenge submission authors.

## Prizes

We will have the following prizes for the winning entries!

- First place: $1,500 cash prize, presentation at ECCV workshop.
- Second place: $500 cash prize, presentation at ECCV workshop.
- Honorable mention(s): $250 cash prize, presentation at ECCV workshop.

## Protocol

- **Evaluation data**: The ECCV '20 challenge evaluates multi-object tracking
  on the TAO test set.

- **Training data**: We do not impose any restrictions on the training data used for
  submissions, except that the TAO test videos may not be used for training in any way.
  This explicitly precludes, for example, unsupervised training on the TAO test set.
  However, the TAO validation videos may be used for training in a supervised or 
  unsupervised manner.
  We encourage training on the LVIS v0.5 dataset, which provides
  ample detection training data for categories evaluated in TAO.

    - **WARNING**: The TAO test set contains sequences from existing datasets, which
        must be excluded from training. These sequences can be seen from the test
        json. In particular, a number of LaSOT training sequences are present in the TAO
        test set.

- For submission instructions, see [evaluation.md](evaluation.md).


## FAQs

Please see [faqs.md](./faqs.md).
