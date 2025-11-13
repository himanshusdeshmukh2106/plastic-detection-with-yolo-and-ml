"""
Train ensemble of models with different random seeds
Expected improvement: +3-5% accuracy over single model
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
from pathlib import Path
import json

# Configuration
BASE_CONFIG = {
    'dataset_path': 'training_dataset/training_dataset',
    'img_size': (224, 224),
    'batch_size': 16,
    'epochs': 100,
    'initial_lr': 0.0005,
    'model_name': 'mobilenet',
    'use_class_weights': True,
}

# Auto-detect dataset path
if not Path(BASE_CONFIG['dataset_path']).exists():
    possible_paths = ['training_dataset', '../training_dataset/training_dataset']
    for path in possible_paths:
        if Path(path).exists() and any(Path(path).iterdir()):
            BASE_CONFIG['dataset_path'] = path
            break

print("=" * 80)
print("ENSEMBLE TRAINING - MULTIPLE MODELS")
print("=" * 80)
print(f"\nBase Configuration:")
for key, value in BASE_CONFIG.items():
    print(f"  {key}: {value}")
print("=" * 80)


def create_data_generators(config):
    """Create data generators"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        config['dataset_path'],
        target_size=config['img_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        config['dataset_path'],
        target_size=config['img_size'],
        batch_size=config['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def calculate_class_weights(train_generator, config):
    """Calculate class weights"""
    class_counts = {}
    for class_name, class_idx in train_generator.class_indices.items():
        class_dir = Path(config['dataset_path']) / class_name
        count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
        class_counts[class_idx] = count
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[class_idx] = weight
    
    return class_weights


def create_model(num_classes, config):
    """Create MobileNetV2 model"""
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(*config['img_size'], 3)
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*config['img_size'], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


def train_single_model(model_id, seed, config, class_weights):
    """Train a single model with given seed"""
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL {model_id} (seed={seed})")
    print("=" * 80)
    
    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Create output directory
    output_dir = f'tf_files_ensemble/model_{model_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data generators
    train_gen, val_gen = create_data_generators(config)
    
    # Create model
    model, base_model = create_model(len(train_gen.class_indices), config)
    
    print(f"\n🏗️  Model {model_id} created with {model.count_params():,} parameters")
    
    # Phase 1: Train classification head
    print(f"\n📚 PHASE 1: Training classification head (base frozen)")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['initial_lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    callbacks_phase1 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_phase1.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks_phase1,
        class_weight=class_weights if config['use_class_weights'] else None,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print(f"\n📚 PHASE 2: Fine-tuning (unfreezing last layers)")
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['initial_lr'] / 20),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    callbacks_phase2 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'best_phase2.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks_phase2,
        class_weight=class_weights if config['use_class_weights'] else None,
        verbose=1
    )
    
    # Evaluate
    results = model.evaluate(val_gen, verbose=0)
    
    print(f"\n📊 Model {model_id} Results:")
    print(f"  Accuracy: {results[1]*100:.2f}%")
    print(f"  Top-2 Accuracy: {results[2]*100:.2f}%")
    
    # Save final model
    model.save(os.path.join(output_dir, 'final_model.h5'))
    print(f"  ✅ Saved to: {output_dir}/final_model.h5")
    
    return model, results, history1, history2


def train_ensemble(num_models=3, seeds=None):
    """Train ensemble of models"""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011][:num_models]
    
    print(f"\n🎯 Training ensemble of {num_models} models")
    print(f"   Seeds: {seeds}")
    
    # Create ensemble directory
    os.makedirs('tf_files_ensemble', exist_ok=True)
    
    # Calculate class weights once
    config = BASE_CONFIG.copy()
    train_gen, _ = create_data_generators(config)
    class_weights = calculate_class_weights(train_gen, config)
    
    print("\n⚖️  Class weights:")
    for class_name, class_idx in train_gen.class_indices.items():
        print(f"  {class_name}: {class_weights[class_idx]:.3f}")
    
    # Save class indices
    with open('tf_files_ensemble/class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    
    # Train each model
    models = []
    results_list = []
    
    for i, seed in enumerate(seeds, 1):
        model, results, hist1, hist2 = train_single_model(i, seed, config, class_weights)
        models.append(model)
        results_list.append(results)
    
    # Summary
    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 80)
    
    print(f"\n📊 Individual Model Results:")
    for i, results in enumerate(results_list, 1):
        print(f"  Model {i}: {results[1]*100:.2f}% accuracy")
    
    avg_accuracy = np.mean([r[1] for r in results_list])
    std_accuracy = np.std([r[1] for r in results_list])
    
    print(f"\n📈 Ensemble Statistics:")
    print(f"  Average accuracy: {avg_accuracy*100:.2f}%")
    print(f"  Std deviation: {std_accuracy*100:.2f}%")
    print(f"  Min accuracy: {min([r[1] for r in results_list])*100:.2f}%")
    print(f"  Max accuracy: {max([r[1] for r in results_list])*100:.2f}%")
    
    # Expected ensemble performance
    expected_ensemble = avg_accuracy + (0.03 if num_models >= 3 else 0.02)
    print(f"\n🎯 Expected ensemble accuracy: {expected_ensemble*100:.2f}%")
    print(f"   (Individual avg + 3% boost from ensemble)")
    
    print(f"\n✅ All models saved in: tf_files_ensemble/")
    print(f"   - model_1/final_model.h5")
    print(f"   - model_2/final_model.h5")
    print(f"   - model_3/final_model.h5")
    
    return models, results_list


if __name__ == '__main__':
    import sys
    
    # Get number of models from command line
    num_models = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    
    print(f"\n🚀 Starting ensemble training with {num_models} models...")
    print(f"⏱️  Estimated time: {num_models * 40} - {num_models * 50} minutes")
    print(f"💾 Disk space needed: ~{num_models * 15} MB")
    
    response = input(f"\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Train ensemble
    models, results = train_ensemble(num_models=num_models)
    
    print("\n" + "=" * 80)
    print("🎉 ENSEMBLE TRAINING COMPLETED!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Test ensemble: python predict_ensemble.py test_image.jpg")
    print("  2. Compare with single model")
    print("  3. Deploy the ensemble for production")
