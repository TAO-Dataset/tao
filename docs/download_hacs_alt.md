Download and extract from YouTube.

```
python scripts/download/download_hacs.py $TAO_ROOT --split train
```

You can ignore YoutubeDL errors that are printed by this script (e.g., Video not
available). Videos that could not be downloaded will be collected in
`$TAO_ROOT/hacs_missing/missing.txt`. You can request the original HACS videos
by filling out these forms: https://forms.gle/hZD612H5TXDQDozv9
