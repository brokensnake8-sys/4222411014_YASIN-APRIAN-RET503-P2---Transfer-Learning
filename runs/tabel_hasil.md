| Mode | Pendekatan | Akurasi val terbaik | Akurasi train akhir | Waktu latih (dtk) | Epoch pertama val >= 90% |
|---|---|---|---|---|---|
| feature | Feature extraction (backbone beku) | 100.00% | 100.00% | 188 | 1 |
| partial | Fine-tuning parsial (blok akhir) | 99.31% | 100.00% | 219 | 1 |
| scratch | Scratch (tanpa pretrained) | 24.05% | 99.36% | 377 | tidak tercapai |
