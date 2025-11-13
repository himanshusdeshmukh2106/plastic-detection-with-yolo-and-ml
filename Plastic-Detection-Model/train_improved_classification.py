"""
Improved training script for Plastic Detection Classification Model
Uses modern techniques and handles class imbalance
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, MobileNetV2, ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
from pathlib import Path

# Configuration
CONFIG = {
    'dataset_path': 'training_dataset/training_dataset',
    'img_size': (224, 224),  # Standard size for transfer learning
    'batch_size': 32,
    'epochs': 50,
    'initial_lr': 0.001,
    'model_name': 'efficientnet',  # 'efficientnet', 'mobilenet', or 'resnet'
    'use_class_weights': True,  # Handle class imbalance
    'output_dir': 'tf_files_improved',
}

# Auto-detect dataset path if default doesn't exist
from pathlib import Path
if not Path(CONFIG['dataset_path']).exists():
    possible_paths = ['training_dataset', '../training_dataset/training_dataset']
    for path in possible_paths:
        if Path(path).exists() and any(Path(path).iterdir()):
            CONFIG['dataset_path'] = path
            print(f"📁 Using dataset path: {path}")
            break

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("=" * 80)
print("IMPROVED PLASTIC DETECTION TRAINING")
print("=" * 80)
print(f"\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print("=" * 80)


def create_data_generators():
    """
    Create data generators with augmentation
    """
    print("\n📊 Creating data generators...")
    
    # Training data augmentation (aggressive)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        CONFIG['dataset_path'],
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        CONFIG['dataset_path'],
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"  ✅ Training samples: {train_generator.samples}")
    print(f"  ✅ Validation samples: {val_generator.samples}")
    print(f"  ✅ Number of classes: {len(train_generator.class_indices)}")
    print(f"  ✅ Classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator


def calculate_class_weights(train_generator):
    """
    Calculate class weights to handle imbalance
    """
    print("\n⚖️  Calculating class weights...")
    
    # Count samples per class
    class_counts = {}
    for class_name, class_idx in train_generator.class_indices.items():
        class_dir = Path(CONFIG['dataset_path']) / class_name
        count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
        class_counts[class_idx] = count
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[class_idx] = weight
        class_name = list(train_generator.class_indices.keys())[class_idx]
        print(f"  {class_name}: {count} samples, weight={weight:.3f}")
    
    return class_weights


def create_model(num_classes):
    """
    Create model with transfer learning
    """
    print(f"\n🏗️  Building model: {CONFIG['model_name']}...")
    
    # Choose base model
    if CONFIG['model_name'] == 'efficientnet':
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(*CONFIG['img_size'], 3)
        )
        print("  Using EfficientNetB3 (best accuracy)")
    elif CONFIG['model_name'] == 'mobilenet':
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(*CONFIG['img_size'], 3)
        )
        print("  Using MobileNetV2 (fast inference)")
    else:  # resnet
        base_model = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(*CONFIG['img_size'], 3)
        )
        print("  Using ResNet50V2 (balanced)")
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*CONFIG['img_size'], 3))
    
    # Preprocessing
    x = inputs
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    print(f"  ✅ Model created with {model.count_params():,} parameters")
    
    return model, base_model


def train_model(model, base_model, train_gen, val_gen, class_weights=None):
    """
    Train the model in two phases
    """
    print("\n" + "=" * 80)
    print("PHASE 1: TRAINING CLASSIFICATION HEAD (Base Frozen)")
    print("=" * 80)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['initial_lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(CONFIG['output_dir'], 'best_model_phase1.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train phase 1
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weights if CONFIG['use_class_weights'] else None,
        verbose=1
    )
    
    print("\n" + "=" * 80)
    print("PHASE 2: FINE-TUNING (Unfreezing Last Layers)")
    print("=" * 80)
    
    # Unfreeze last layers of base model
    base_model.trainable = True
    
    # Freeze all layers except last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    print(f"  Unfrozen layers: {sum([1 for layer in base_model.layers if layer.trainable])}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['initial_lr'] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    # Update callbacks
    callbacks[2] = ModelCheckpoint(
        os.path.join(CONFIG['output_dir'], 'best_model_phase2.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train phase 2
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights if CONFIG['use_class_weights'] else None,
        verbose=1
    )
    
    return history1, history2


def evaluate_model(model, val_gen):
    """
    Evaluate final model
    """
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    results = model.evaluate(val_gen, verbose=0)
    
    print(f"\n📊 Validation Results:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]*100:.2f}%")
    print(f"  Top-2 Accuracy: {results[2]*100:.2f}%")
    
    # Per-class accuracy
    print(f"\n📋 Per-Class Performance:")
    predictions = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    
    class_names = list(val_gen.class_indices.keys())
    
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean()
            print(f"  {class_name:<15} Accuracy: {class_acc*100:>6.2f}%")
    
    return results


def save_model(model, train_gen):
    """
    Save model in multiple formats
    """
    print("\n💾 Saving model...")
    
    # Save in Keras format
    model.save(os.path.join(CONFIG['output_dir'], 'final_model.h5'))
    print(f"  ✅ Saved: {CONFIG['output_dir']}/final_model.h5")
    
    # Save labels
    labels_path = os.path.join(CONFIG['output_dir'], 'labels.txt')
    with open(labels_path, 'w') as f:
        for class_name in train_gen.class_indices.keys():
            f.write(f"{class_name}\n")
    print(f"  ✅ Saved: {labels_path}")
    
    # Save as TensorFlow SavedModel
    model.save(os.path.join(CONFIG['output_dir'], 'saved_model'))
    print(f"  ✅ Saved: {CONFIG['output_dir']}/saved_model")
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    # Calculate class weights
    class_weights = None
    if CONFIG['use_class_weights']:
        class_weights = calculate_class_weights(train_gen)
    
    # Create model
    model, base_model = create_model(len(train_gen.class_indices))
    
    # Train model
    history1, history2 = train_model(model, base_model, train_gen, val_gen, class_weights)
    
    # Evaluate model
    results = evaluate_model(model, val_gen)
    
    # Save model
    save_model(model, train_gen)
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nFinal Validation Accuracy: {results[1]*100:.2f}%")
    print(f"\nModel saved to: {CONFIG['output_dir']}/")
    print("\nNext steps:")
    print("  1. Test the model with: python test_improved_model.py")
    print("  2. Compare with original model")
    print("  3. Deploy to production")
