# Download TAO

TAO contains videos from 7 datasets: ArgoVerse, AVA, BDD-100k, Charades, HACS,
LaSOT, and YFCC-100M.

1. Download TAO train, val and test sets from the MOTChallenge
   [download page](https://motchallenge.net/tao_download.php).

1. Uncompress each downloaded file in a single directory, which we will refer to as
   `$TAO_DIR`.
   <details><summary>The directory should have the following structure:</summary><p>

   ```bash
    └── frames
       └── train
          ├── ArgoVerse
          ├── BDD
          ├── Charades
          ├── HACS
          ├── LaSOT
          └── YFCC100M
    ```
    </p></details>

1. Download annotations:

    ```bash
    python scripts/download/download_annotations.py $TAO_DIR --split train
    ```

1. Verify that the dataset was downloaded correctly:

    ```bash
    python scripts/download/verify.py $TAO_ROOT --split train
    ```
## Request video deletion

If you would like to request a video be deleted from TAO (e.g., because you are
featured in the video or you own the rights), please email me at
achald@cs.cmu.edu.

