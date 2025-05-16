import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from typing import Any, List, Optional
import nltk

nltk.download("averaged_perceptron_tagger_eng")

# Global variable for ProcessPoolExecutor usage.
augmenters = {}


def _init_worker(random_state: int):
    np.random.seed(random_state)
    global augmenters
    augmenters = {
        "synonym": naw.SynonymAug(aug_src="wordnet"),
        "contextual": naw.ContextualWordEmbsAug(
            model_path="distilbert-base-uncased", action="substitute"
        ),
        "random": naw.RandomWordAug(action="swap"),
        "sentence_shuffle": nas.RandomSentAug(),
        "word_deletion": naw.RandomWordAug(action="delete", aug_p=0.1),
    }


class DataAugmentation:

    def __init__(
        self, random_state: int = 42, num_workers: Optional[int] = None
    ) -> None:

        self.random_state = random_state
        self.num_workers = num_workers or os.cpu_count()

    @staticmethod
    def augment_text(text: str, methods: List[str], n: int) -> List[str]:

        augmented = []
        for method in methods:
            augmenter = augmenters.get(method)
            if not augmenter:
                continue
            try:
                out = augmenter.augment(text, n=n)
            except Exception:
                continue
            if isinstance(out, str):
                out = [out]
            augmented.extend([o for o in out if o])

        return augmented

    @staticmethod
    def _augment_worker(args: Any) -> List:

        row_dict, text_col, methods, n = args
        texts = DataAugmentation.augment_text(row_dict[text_col], methods, n)
        new_rows = []
        for txt in texts[:n]:
            nr = row_dict.copy()
            nr[text_col] = txt
            new_rows.append(nr)

        return new_rows

    def balance_dataset(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        target_counts: Optional[dict] = None,
        methods: List[str] = None,
    ) -> pd.DataFrame:

        class_counts = df[label_col].value_counts().to_dict()
        max_count = max(class_counts.values())

        if target_counts is None:
            target_counts = {lab: int(max_count * 0.8) for lab in class_counts}

        balanced = df.copy()
        methods = methods or list(augmenters.keys())

        all_augmented = []
        init_args = (self.random_state,)

        for label, cnt in class_counts.items():

            target = target_counts.get(label, max_count)
            if cnt >= target:
                continue
            needed = target - cnt
            samples = df[df[label_col] == label]
            aug_per = int(np.ceil(needed / len(samples)))

            task_args = [
                (row.to_dict(), text_col, methods, aug_per)
                for _, row in samples.iterrows()
            ]

            new_entries = []
            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_worker,
                initargs=init_args,
            ) as executor:
                futures = [
                    executor.submit(self._augment_worker, arg) for arg in task_args
                ]
                for fut in as_completed(futures):
                    try:
                        new_entries.extend(fut.result())
                    except Exception as e:
                        print(f"Worker error for label {label}: {e}")

            new_rows = new_entries[:needed]
            if new_rows:
                balanced = pd.concat(
                    [balanced, pd.DataFrame(new_rows)], ignore_index=True
                )
                all_augmented.extend(new_rows)

        if all_augmented:
            aug_df = pd.DataFrame(all_augmented)
            self.save_augmented_data(aug_df, "augmented.csv")

        return balanced

    @staticmethod
    def save_augmented_data(df: pd.DataFrame, file_path: str) -> None:

        df.to_csv(file_path, index=False)
        print(f"Saved {len(df)} augmented samples to {file_path}")
