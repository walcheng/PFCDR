# Privacy-Friendly Cross-Domain Recommendation via Distilling User-irrelevant Information (PFCDR)

## Introduction
This repository provides the implementations of PFCDR and two popular baselines (TGTOnly, EMCDR):
* TGTOnly：Train a MF model with the data of the target domain.
* EMCDR: [Cross-Domain Recommendation: An Embedding and Mapping Approach](https://www.ijcai.org/Proceedings/2017/0343.pdf) (IJCAI 2017)



## Requirements

- Python 3.6
- Pytorch > 1.0
- Pandas
- Numpy
- Tqdm

## File Structure

```
.
├── code
│   ├── config.json             # Common Configurations
│   ├── inversion_config.json   # Configurations for Inversion stage
│   ├── main.py                 # Entry function
│   ├── models.py               # Models based on MF, GMF or Youtube DNN
│   ├── preprocessing.py        # Parsing and Segmentation
│   ├── inversion.py            # Module for model inversion
│   ├── float_embedding.py      # A new version of torch.nn.embedding that support float as input
│   ├── README.md
│   └── run.py                  # Training and Evaluating 
└── data
    ├── mid                 # Mid data
    │   ├── Books.csv
    │   ├── CDs_and_Vinyl.csv
    │   └── Movies_and_TV.csv
    ├── raw                 # Raw data
    │   ├── reviews_Books_5.json.gz
    │   ├── reviews_CDs_and_Vinyl_5.json.gz
    │   └── reviews_Movies_and_TV_5.json.gz
    └── ready               # Ready to use
        ├── _2_8
        ├── _5_5
        └── _8_2
```

## Dataset

We utilized the Amazon Reviews 5-score dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./data/ready`.

```python
python main.py --process_data_mid 1 --process_data_ready 1
```

## Run

Parameter Configuration:

- task: different tasks within `1, 2 or 3`, default for `1`
- base_model: different base models within `MF, GMF or DNN`, default for `MF`
- ratio: train/test ratio within `[0.8, 0.2], [0.5, 0.5] or [0.2, 0.8]`, default for `[0.8, 0.2]`
- epoch: pre-training and CDR mapping training epoches, default for `10`
- seed: random seed, default for `2020`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.01`
- model_name: base model for embedding, default for `MF`

You can run this model through:

```powershell
# Run directly with default parameters 
python main.py --process_data_mid 0 --process_data_ready 0

```
It will run a little while since it need to generate prototypes on overlapped users in source and target domains. This 
prototypes will automatically save, next time you can directly use some users for CDR.

If you wanna try different `weight decay`, `embedding dimmension` or more tasks, you may change 
the settings in `./config.json`. If you wanna try the parameter of inversion stage you may change 
the settings in `./inversion_config.json`.

If the code is helpful for you, please cite this paper:

```
@inproceedings{wang2025privacy,
  title={Privacy-Friendly Cross-Domain Recommendation via Distilling User-irrelevant Information},
  author={Wang, Cheng and Xu, Wenchao and Wang, Haozhao and Liu, Wei and Li, Ruixuan},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={450--461},
  year={2025}
}
```
