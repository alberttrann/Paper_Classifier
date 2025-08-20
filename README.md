First, we have to run prepare_data_and_faiss.py: 

Since the full train split of the dataset contains 2.29 million rows, it is risky to process the whole thing all at once because it might exceed the available RAM

That's why `prepare_data_and_faiss.py` implements a chunk-based embedding generation. The `run_ensemble_classification.py` script will simply load the complete, concatenated embedding array once it's saved to disk.

1.  **`TOTAL_DATASET_SIZE_STR = "train[:2290000]"`:**
    *   This explicitly tells `load_dataset` to load up to 2.29 million samples from the `train` split. If you want the *entire* dataset regardless of its exact reported size, you could use `"train"`. I've used `[:2290000]` for clarity, but be aware that Hugging Face `datasets` might still try to load the full dataset if it's not truly streamable for a direct `len()` call.
    *   **`streaming=True`:** This is crucial. `load_dataset(..., streaming=True)` allows you to iterate over the dataset without loading the entire thing into RAM at once, which is perfect for processing large files in chunks.

2.  **`CHUNK_SIZE = 458000`:**
    *   This defines the number of abstracts to process for embeddings at a time.
    *   For 2.29M rows, `2290000 / 458000 = 5` chunks. Each chunk of embeddings will be roughly 1.4 GB, which is very manageable for your 16GB RAM.

3.  **Chunked Processing Loop:**
    *   The script now iterates through the `dataset_generator` (which streams the data).
    *   It accumulates `current_chunk_abstracts` and `current_chunk_labels` until `CHUNK_SIZE` is reached.
    *   Once a chunk is full, it generates embeddings for *only that chunk*.
    *   The generated embeddings for the chunk (`embeddings_chunk`) are then immediately saved to a temporary `.npy` file on disk.
    *   `current_chunk_abstracts` and `current_chunk_labels` are cleared, and `embeddings_chunk` is explicitly deleted (`del embeddings_chunk`) to free up RAM. This is the core of the memory optimization.
    *   `tqdm` is used to show progress for both reading the dataset and for generating embeddings within each chunk.

4.  **Concatenation from Disk:**
    *   After all chunks are processed and saved, the script then iterates through the list of `temp_chunk_files`.
    *   It loads each chunk from disk *one by one* and appends it to `full_embeddings_list`.
    *   Finally, `np.vstack` concatenates all loaded chunks into a single `full_embeddings` NumPy array, which is then used for the train/val/test split and FAISS indexing.
    *   **Crucially:** The temporary chunk files are deleted at the end to clean up your disk space.

### How to Run:

1.  **Ensure you have enough disk space!** 2.29M embeddings (7GB) plus temporary chunk files will require more than 7GB during the process (e.g., 7GB for the final array + 1.4GB for a chunk * 2 if one is being written while another is being read/generated). Ensure you have at least 15-20 GB free.
2.  **Run `prepare_data_and_faiss.py`:**
    ```bash
    python prepare_data_and_faiss.py
    ```
    This will take a significant amount of time (many hours) due to processing the full dataset and generating embeddings, but it should remain within your 16GB RAM limit for active processing thanks to chunking.

    After complete, ,things will loo something like this:

    ```
    Finished generating embeddings for a total of 2292057 samples across 6 chunks.
    Finished generating embeddings for a total of 2292057 samples across 6 chunks.
    Concatenating all L2-normalized embedding chunks from disk...
    Loading and concatenating chunks: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:16<00:00,  2.68s/it] 
    Concatenated embeddings shape: (2292057, 768)
    Temporary chunk files removed.
    Mapping string labels to integers...
    Mapping Labels: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 2292057/2292057 [00:00<00:00, 2784705.71it/s] 
    Total samples with embeddings and integer labels: 2292057
    Checking for and handling singleton classes...
    Original samples: 2292057. Samples after filtering singletons: 2245440
    Removed 46617 samples belonging to singleton classes.
    Splitting data into training, validation, and test sets...
    Train set shape: (1571808, 768), Labels: (1571808,)
    Validation set shape: (336816, 768), Labels: (336816,)
    Test set shape: (336816, 768), Labels: (336816,)
    Building FAISS IndexFlatIP index for training embeddings...
    FAISS index built. Number of vectors in index: 1571808
    Saving processed data to 'processed_data'...
    Data preparation and FAISS index creation complete.
    ```

3.  **Run `run_ensemble_classification.py`:**
    ```bash
    python run_ensemble_classification.py
    ```
    This script will then load the full, pre-generated `full_embeddings.npy` file and proceed with the ensemble training and evaluation.

    

