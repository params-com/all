# Recommender System Template (Keras 3)

A flexible recommender system template using **Keras 3** that can be customized for various recommendation tasks. The system implements collaborative filtering with neural embeddings and supports optional user and item features.

---

## Table of contents

* [Shape of the problem](#shape-of-the-problem)
* [Key inputs and outputs](#key-inputs-and-outputs)
* [Usage](#usage)
* [Configuration](#configuration)
* [Directory structure](#directory-structure)
* [Approach](#approach)

  * [Model architecture](#model-architecture)
  * [Dataset preparation](#dataset-preparation)
  * [Training strategy](#training-strategy)
* [Dummy data (generator)](#dummy-data-generator)
* [Compatibility](#compatibility)
---

## Shape of the problem

We consider a fixed set of **users** and a fixed set of **items** (products, songs, games, ...). The dataset contains **interactions** between users and items (for example, a user listened to a song or rated a product). Interactions can optionally include a numeric score (rating), but the template supports:

* binary interactions only (implicit feedback), or
* interactions with scalar ratings (explicit feedback).

The model learns dense embeddings for users and items. The **dot product** between a user embedding and an item embedding produces an affinity score that indicates how likely the user is to interact with the item in the future.

Both users and items may have side features (age, gender, genres, price, etc.). The encoders can incorporate those features — if no features are available the model still works using only ID embeddings.

---

## Key inputs and outputs

### Inputs

* Interaction data with fields:

  * `user` (identifier)
  * `item` (identifier)
  * optional: `score` / `rating`
* Optional user features (tabular / categorical / numeric)
* Optional item features (tabular / categorical / numeric)

### Outputs

* Trained recommendation model (Keras saved model / weights)
* `recommendations.json` — top-N recommendations per user (excludes previously interacted items)
* Optional visualizations saved to `figures/` (if `exploratory_data_analysis` is run)

---

## Usage

1. Install dependencies. Choose the backend/runtime matching your hardware:

```bash
# For JAX GPU
pip install -r requirements-jax.txt

# For PyTorch GPU
pip install -r requirements-torch.txt

# For TensorFlow GPU
pip install -r requirements-tensorflow.txt
```

2. Configure essential settings in `config.py`:

* `raw_interaction_data_fpath`, `raw_user_data_fpath`, `raw_item_data_fpath` — point these to your CSV/JSON files.
* `min_interactions_per_user`, `min_interactions_per_item` — data cleaning thresholds.
* Model/training parameters (embedding dimension, batch size, learning rate, etc.).

3. Implement `get_raw_data()` in `data.py` so it correctly parses your raw data and returns the expected tuple:

```py
# returns
interaction_data: List[dict]  # each dict contains: {"user": ..., "item": ..., optional: "score": ...}
user_data: Dict               # optional, keyed by user id
item_data: Dict               # optional, keyed by item id
```

4. Run one of the main actions:

```bash
python main.py exploratory_data_analysis        # run EDA and save figures
python main.py train_validation_model          # train and evaluate on a held-out validation set
python main.py train_production_model           # train final model on full dataset
python main.py compute_predictions             # produce recommendations for all users -> recommendations.json
```

---

## Configuration

Key configuration is contained in `config.py`. Typical options:

* Data paths: `raw_interaction_data_fpath`, `raw_user_data_fpath`, `raw_item_data_fpath`
* Filtering: `min_interactions_per_user`, `min_interactions_per_item`
* Modeling: `embedding_dim`, `user_feature_dims`, `item_feature_dims`
* Training: `batch_size`, `learning_rate`, `epochs`, `early_stopping_patience`
* Output: `model_dir`, `recommendations_path`, `figures_dir`

---

## Directory structure

```
├─ config.py
├─ data.py                # data loading + preprocessing
├─ eda.py                 # exploratory data analysis
├─ models.py              # model architecture (Encoders, EmbeddingModel)
├─ train.py               # training loop, callbacks, evaluation
├─ main.py                # main entrypoint (cli/arg dispatch)
├─ dummy_data/            # synthetic dataset generator + sample CSVs
│  ├─ users.csv
│  ├─ games.csv
│  └─ user_ratings.csv
├─ figures/               # generated EDA figures
└─ recommendations.json   # output of compute_predictions
```

---

## Approach

### Model architecture

The model uses a **two-tower neural architecture**:

* **User encoder** (class `Encoder`): combines an ID embedding for the user with any optional user features (categorical embeddings, numeric inputs, small MLP) and outputs a fixed-length user embedding vector.
* **Item encoder** (class `Encoder`): symmetric to the user encoder — combines item ID embedding with item features to produce an item embedding vector.
* **EmbeddingModel**: applies both encoders and computes the affinity between user and item via the dot product of their embeddings. This affinity can be used directly (for ranking) or passed through an activation for specific losses.

This setup is modular so you can swap encoder internals (e.g., add more MLP layers, interaction layers, feature cross layers) easily.

### Dataset preparation

* The `InteractionDataset` (a Keras `PyDataset`) provides batches of training data.
* Each batch contains **positive samples** (observed interactions) and **negative samples** (sampled items the user did not interact with). Negative sampling can be uniform or popularity-biased.
* Labels for training depend on data type:

  * Implicit data: binary labels (1 for positive, 0 for negative) with a binary loss (e.g., `binary_crossentropy` or ranking loss).
  * Explicit ratings: regression or classification targets, or converted into pairwise losses.

### Training strategy

* Use callbacks to improve robustness:

  * `EarlyStopping` to prevent overfitting.
  * `ModelCheckpoint` to save best model weights.
* Optionally use learning rate schedules, gradient clipping, and mixed precision (if hardware supports it).
* Evaluate with ranking metrics on the validation set: Precision\@K, Recall\@K, NDCG\@K, MAP, or RMSE if using ratings.

---

## Expected training time

Training time depends heavily on dataset size, embedding dimension, and hardware.

* **Small datasets (<100K interactions)**: minutes–hours on CPU.
* **Medium (100K–1M interactions)**: hours; GPU recommended.
* **Large (>1M interactions)**: hours–days; GPU required for practical speeds.

Tune `batch_size`, `embedding_dim`, and `num_negative_samples` to balance memory/time trade-offs.

---

## Dummy data

A synthetic dataset generator is included to help test and demonstrate the pipeline. It generates realistic gaming data (users, games, user-game ratings) with configurable biases.

### Files (in `dummy_data/`):

* `users.csv`:

  * `id` — unique user identifier
  * `age` — integer (13–80)
  * `gender` — `M` or `F`

* `games.csv`:

  * `id` — unique game identifier
  * `category` — (RPG, FPS, Strategy, Sports, Puzzle, Adventure, Simulation, Fighting, Platform, Racing)
  * `difficulty` — (Easy, Medium, Hard, Expert)
  * `created_at_ms` — creation timestamp in milliseconds

* `user_ratings.csv`:

  * `user_id` — foreign key to users
  * `game_id` — foreign key to games
  * `rating` — 1–5
  * `timestamp_ms` — timestamp in milliseconds

### Generation details

* The generator adds realistic patterns and biases:

  * Age influences category preferences (e.g., younger users skew toward FPS / Fighting)
  * Gender influences category preferences for some categories
  * Age groups show different difficulty preferences
  * Ratings include noise to simulate real-world variability

Default generation sizes:

* `50,000` users
* `5,000` games
* `500,000` ratings
* Data spans 365 days

You can customize the generation via `GameGeneratorConfig` in the repository.

---

## Compatibility

* Python 3.10+
* Keras 3.7+ (works with TensorFlow, JAX, or PyTorch backends)
* Supports CPU and GPU training

---


## License

MIT 


