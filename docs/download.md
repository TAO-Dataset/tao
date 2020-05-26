# Download TAO

TAO contains videos from 7 datasets: ArgoVerse, AVA, BDD-100k, Charades, HACS,
LaSOT, and YFCC-100M.
Where possible, we share the videos from these datasets, and provide scripts to
convert them to frames.
For ArgoVerse and LaSOT, the original datasets are shared as JPEG frames, so we
share those directly.
For HACS and AVA, we share a script to download the data directly from YouTube
and CVDF, respectively.

We are actively working on simplifying these steps.

1. Download videos or frames for each source dataset:

    |           | Train | Val | Test |
    |-----------|-------|-----|------|
    | Charades  | [videos](https://drive.google.com/open?id=12dCphuV-ByVvktkuYFjYl_ZJ1ZVonBjy) | [videos](https://drive.google.com/open?id=1cjrxbf-wA8fxJ2AiHZeLmACZrqF4uPKb) | Coming soon |
    | BDD-100K  | [videos](https://drive.google.com/open?id=191QRs4O11EKHnDoscWws2g8BPiT35yZC) | [videos](https://drive.google.com/open?id=1zpoYCSU4SxIu35tZ1QpfZPpYMu8Kz2o2) | Coming soon |
    | YFCC-100M | [videos](https://drive.google.com/open?id=1j6hdIIFIBwRUthe2kozdB5QvcoBL_-eS) | [videos](https://drive.google.com/open?id=1Eh81fksBo5YwslblsRYib6-Dmy0KfF2k) | Coming soon |
    | LaSOT     | [frames](https://drive.google.com/open?id=1PVW7PrdAVMbX11pmkLkxst-GJPPQai0m) | [frames](https://drive.google.com/open?id=1efg_CV7TgkAQNQWzLExWYHqiHc_ijtF5) | Coming soon |
    | ArgoVerse | [frames](https://drive.google.com/open?id=1RPKIuOEjnTLxSf8qlarkzeS16G7si7bu) | [frames](https://drive.google.com/open?id=1zwhwDifJ5YW2F42e4PLeJVTvZ8W2dq-Q) | Coming soon |

    NOTE: The rest of these instructions assume you are downloading the `train`
    split. Simply replace `train` with `val` everywhere to run on `val`.

1. Download HACS videos.

    1. Option 1: Please request a download URL by filling out this form:
       https://forms.gle/hZD612H5TXDQDozv9 . We will send you the URL within 24
       hours.
    1. Alternatively, you can download the videos directly from YouTube by
       following instructions [here](./download_hacs_alt.md).

1. Uncompress each file in a single directory, which we will refer to as
   `$TAO_DIR`.
   <details><summary>The directory should have the following structure:</summary><p>

   ```bash
    ├── frames
    │  └── train
    │     ├── ArgoVerse
    │     └── LaSOT
    └── videos
        └── train
            ├── BDD
            ├── Charades
            ├── HACS
            └── YFCC100M
    ```
    </p></details>

1. Run the following helper command to handle the rest of the download process.

    ```bash
    python scripts/download/download_helper.py $TAO_ROOT --split train
    ```

    This command will do the following:
    - Download annotations
    - Extract frames from BDD, Charades, HACS, and YFCC-100M videos.
    - Download AVA movies and extract frames.

    If you run into issues with this command, you can also run each step
    manually, following the directions [here](./manual_download.md).

## Request video deletion

If you would like to request a video be deleted from TAO (e.g., because you are
featured in the video or you own the rights), please email me at
achald@cs.cmu.edu.

