# Download TAO

TAO contains videos from 7 datasets: ArgoVerse, AVA, BDD-100k, Charades, HACS,
LaSOT, and YFCC-100M.

1. Download frames for ArgoVerse, BDD, Charades, LaSOT and YFCC datasets:

    |           | Train | Val | Test |
    |-----------|-------|-----|------|
    | Charades  | [frames](https://drive.google.com/open?id=1p7X7nZTBA0GinkGikyN4xb0zp3UOEyTY) | [frames](https://drive.google.com/open?id=1vdll7eqSXrfs0TKdzn4OjCYgFPfavrJO) | [frames](https://drive.google.com/open?id=1QhpcUsRcHeCnxspUuR4oTLIAoW_nU2ca) |
    | BDD-100K  | [frames](https://drive.google.com/open?id=1p8Ai2QqNqWPQ1A-7U_KaB24mYWXZxMe5) | [frames](https://drive.google.com/open?id=1cm6ktkg64chYHXxR0Gxbo352R1a-RSrj) | [frames](https://drive.google.com/open?id=1GEPIrrb9rE8aYlxOQ6PW_tWKqbDWc4r9)|
    | YFCC-100M | [frames](https://drive.google.com/open?id=1kstavuoql0C307Ndp4QUaQebrfB7sqgO) | [frames](https://drive.google.com/open?id=1Dqh9HTsMNcn-YVG5aG2Ra9YQrgN8eack) | [frames](https://drive.google.com/open?id=1xPUvrAWrv7jrhEjT8PfiducLLGebdEtb) |
    | LaSOT     | [frames](https://drive.google.com/open?id=1PVW7PrdAVMbX11pmkLkxst-GJPPQai0m) | [frames](https://drive.google.com/open?id=1efg_CV7TgkAQNQWzLExWYHqiHc_ijtF5) | [frames](https://drive.google.com/open?id=105ZSIVu8mAQlim907Rj8hRtOUvKZ6cem) |
    | ArgoVerse | [frames](https://drive.google.com/open?id=1RPKIuOEjnTLxSf8qlarkzeS16G7si7bu) | [frames](https://drive.google.com/open?id=1zwhwDifJ5YW2F42e4PLeJVTvZ8W2dq-Q) | [frames](https://drive.google.com/open?id=1rwPcIzp1NaqSTEDf9jPc4mEw4ysqp5MO) |

    NOTE: The rest of these instructions assume you are downloading the `train`
    split. Simply replace `train` with `val` everywhere to run on `val`.

1. Download HACS and AVA frames. Please request a download URL by filling out this form:
   https://forms.gle/hZD612H5TXDQDozv9 . We will send you the URL within 24
   hours.

1. Uncompress each file in a single directory, which we will refer to as
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

