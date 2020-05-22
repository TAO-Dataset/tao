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
    | LaSOT     | [frames](https://drive.google.com/open?id=1L0UUX_H7X1GGNt28v9I-g7UYRU1fHpaR) | [frames](https://drive.google.com/open?id=1UjLW_-0V3niEIyWhKJv0tSxzPaKr8264) | Coming soon |
    | ArgoVerse | [frames](https://drive.google.com/open?id=1jVhCHmaeFXJ6Vzbcn4LR_HqkVvTa2YHN) | [frames](https://drive.google.com/open?id=1iUyNg0XuruCizvcoiYyEUioK6_gThmqv) | Coming soon |

    NOTE: The rest of these instructions assume you are downloading the `train`
    split. Simply replace `train` with `val` everywhere to run on `val`.

1. Uncompress each file in a single directory, which we will refer to as
   `$TAO_DIR`.  The directory should have the following structure:

    ```
    ├── frames
    │  └── train
    │     ├── ArgoVerse
    │     └── LaSOT
    └── videos
        └── train
            ├── BDD
            ├── Charades
            └── YFCC100M
    ```

1. Download TAO annotations to $TAO_DIR

    ```
    wget 'https://github.com/TAO-Dataset/annotations/archive/v1.0.tar.gz'
    tar xzvf v1.0.tar.gz
    mv annotations-v1.0 annotations
    ```

1. Extract frames from BDD, Charades, and YFCC-100M.

    ```
    python scripts/download/extract_frames.py $TAO_ROOT --split train
    ```

    After this, your directory should have the following structure:

    ```
    ├── frames
    │  └── train
    │     ├── ArgoVerse
    │     ├── BDD
    │     ├── Charades
    │     ├── LaSOT
    │     └── YFCC100M
    └── videos
        └── train
            ├── BDD
            ├── Charades
            └── YFCC100M
    ```

1. Download and extract frames from AVA:

    ```
    python scripts/download/download_ava.py $TAO_ROOT --split train
    ```

1. Download and extract frames from HACS:

    1. Download and extract from YouTube.

        ```
        python scripts/download/download_hacs.py $TAO_ROOT --split train
        ```

        You can ignore YoutubeDL errors that are printed by this script (e.g.,
        Video not available). At the end, we will collect a list of all the
        videos that we could not download and handle them in the next step.

    1. Download any HACS videos that were taken down from YouTube from the HACS
        website:

        https://github.com/hangzhaomit/HACS-dataset#request-testing-videos-and-missing-videos-new

        Store these in `$TAO_ROOT/hacs_missing/`

    1. Extract frames from newly downloaded HACS videos by re-running
    `download_hacs.py`.

        ```
        python scripts/download/download_hacs.py $TAO_ROOT --split train
        ```

1. Finally, you should have the following structure

    ```
    ├── frames
    │  └── train
    │     ├── ArgoVerse
    │     ├── AVA
    │     ├── BDD
    │     ├── Charades
    │     ├── HACS
    │     ├── LaSOT
    │     └── YFCC100M
    └── videos
        └── train
            ├── BDD
            ├── Charades
            └── YFCC100M
    ```
