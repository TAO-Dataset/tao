These are alternative instructions that mimic the helper script in
[scripts/download/download_helper.py](/scripts/download/download_helper.py),
in case the helper script causes issues. Please read
[./download.md](./download.md) first.

1. Download TAO annotations to $TAO_DIR

    ```
    wget 'https://github.com/TAO-Dataset/annotations/archive/v1.0.tar.gz'
    tar xzvf v1.0.tar.gz
    mv annotations-v1.0 annotations
    ```

1. Extract frames from BDD, Charades, HACS and YFCC-100M.

    ```
    python scripts/download/extract_frames.py $TAO_ROOT --split train
    ```
   <details><summary>After this, your directory should have the following structure:</summary><p>

    ```
    ├── frames
    │  └── train
    │     ├── ArgoVerse
    │     ├── BDD
    │     ├── Charades
    │     ├── HACS
    │     ├── LaSOT
    │     └── YFCC100M
    └── videos
        └── train
            ├── BDD
            ├── Charades
            ├── HACS
            └── YFCC100M
    ```
    </details>

1. Download and extract frames from AVA:

    ```
    python scripts/download/download_ava.py $TAO_ROOT --split train
    ```

1. Finally, you can verify that you have downloaded TAO.

   <details><summary>Expected directory structure</summary><p>

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
    </details>

    You can run the following command to check that TAO was properly extracted:

    ```
    python scripts/download/verify.py $TAO_ROOT --split train
    ```
