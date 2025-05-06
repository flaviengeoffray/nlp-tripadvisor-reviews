# Configuration parameters
BATCH_SIZE = 16
EMBED_SIZE = 128  # Increased for word-level model
HIDDEN_SIZE = 256  # Increased for word-level model
MAX_SAMPLES = 1000 # 25000  # Limit dataset size for faster training # MAX = 201295
MAX_SEQUENCE_LENGTH = 100  # Max number of words per review
EPOCHS = 10 # Increased epochs for better learning
MIN_WORD_FREQ = 3  # Minimum frequency for a word to be included in vocabulary
LEARNING_RATE = 0.001
