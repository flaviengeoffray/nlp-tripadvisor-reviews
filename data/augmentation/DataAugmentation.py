from datasets import load_dataset
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from typing import List, Optional

import nltk
nltk.download('averaged_perceptron_tagger_eng')

class DataAugmentation:
    """
    Class for augmenting textual data, particularly useful for balancing
    minority classes in an imbalanced dataset.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initializes the data augmentation class.
        
        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize augmenters
        self.augmenters = {
            'synonym': naw.SynonymAug(aug_src='wordnet'),
            'contextual': naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute"),
            'random': naw.RandomWordAug(action="swap"),
            'sentence_shuffle': nas.RandomSentAug(),
            'word_deletion': naw.RandomWordAug(action="delete", aug_p=0.1),
        }
    
    def augment_text(self, text: str, methods: List[str] = None, n: int = 1) -> List[str]:
        """
        Augments a text using multiple methods.
        
        Args:
            text: The text to augment
            methods: List of augmentation methods to use
            n: Number of augmentations per method
            
        Returns:
            List of augmented texts
        """

        # print(f"Original text: {text}\n")
        if methods is None:
            methods = ['synonym', 'random']
        
        augmented_texts = []
        
        for method in methods:
            if method in self.augmenters:
                try:
                    augmenter = self.augmenters[method]
                    augmented = augmenter.augment(text, n=n)
                    # print(f"Augmented text using {method}:\n{augmented}\n")
                    if isinstance(augmented, str):
                        augmented = [augmented]
                    
                    augmented_texts.extend(augmented)
                except Exception as e:
                    print(f"Error during augmentation with method {method}: {e}")
        
        return augmented_texts
    
    def balance_dataset(
        self, 
        df: pd.DataFrame, 
        text_col: str, 
        label_col: str,
        target_counts: Optional[dict] = None, 
        methods: List[str] = None
    ) -> pd.DataFrame:
        """
        Balances a DataFrame by augmenting minority classes.
        
        Args:
            df: DataFrame containing the data
            text_col: Name of the column containing the text
            label_col: Name of the column containing the labels
            target_counts: Dictionary specifying the target number of samples for each class
                          (if None, all classes will be balanced to match the majority class)
            methods: Augmentation methods to use
        Returns:
            Balanced DataFrame
        """

        # Count occurrences of each class
        class_counts = df[label_col].value_counts().to_dict()
        max_count = max(class_counts.values())

        # If target_counts is not provided, set all classes to 80% of the max count
        if target_counts is None:
            target_counts = {label: int(max_count * 0.8) for label in class_counts.keys()}
        
        balanced_df = df.copy()
        
        for label, count in class_counts.items():
            target = target_counts.get(label, max_count)
            
            if count < target:
                # Number of samples to add
                samples_needed = target - count
                
                # Select samples of the current class
                class_samples = df[df[label_col] == label]
                
                # Calculate how many augmentations per sample
                augmentations_per_sample = int(np.ceil(samples_needed / len(class_samples)))
                
                new_samples = []
                
                # For each sample of the minority class
                for _, row in class_samples.iterrows():
                    # Augment the text
                    augmented_texts = self.augment_text(
                        row[text_col], 
                        methods=methods, 
                        n=augmentations_per_sample
                    )
                    
                    # Add the new augmented samples
                    for aug_text in augmented_texts[:min(augmentations_per_sample, samples_needed)]:
                        new_row = row.copy()
                        new_row[text_col] = aug_text
                        new_samples.append(new_row)
                        samples_needed -= 1
                        
                        if samples_needed <= 0:
                            break
                                            
                # Add the new samples to the DataFrame
                if new_samples:
                    balanced_df = pd.concat([balanced_df, pd.DataFrame(new_samples)], ignore_index=True)
                    self.save_augmented_data(pd.DataFrame(new_samples), f"augmented.csv")

        return balanced_df

    def save_augmented_data(self, df: pd.DataFrame, file_path: str):
  
        df.to_csv(file_path, index=False)
        print(f"Augmented data saved to {file_path}")

