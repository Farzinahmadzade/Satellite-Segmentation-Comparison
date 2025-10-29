class Config:
    IMAGE_SIZE = (256, 256)
    NUM_CLASSES = 9
    CLASS_NAMES = [
        "bareland", "rangeland", "development", "road", "tree",
        "water", "agricultural", "building", "nodata"
    ]
    BATCH_SIZE = 4
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2
    NUM_WORKERS = 2