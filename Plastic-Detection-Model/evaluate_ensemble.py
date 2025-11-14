"""
Evaluate the trained ensemble models
"""
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import os
from pathlib import Path

def load_ensemble_models(ensemble_dir='ensemble_models/tf_files_ensemble'):
    """Load all models from ensemble"""
    models = []
    model_dirs = sorted([d for d in Path(ensemble_dir).iterdir() if d.is_dir() and d.name.startswith('model_')])
    
    print(f"Loading {len(model_dirs)} models...")
    
    for model_dir in model_dirs:
        model_path = model_dir / 'final_model.h5'
        if model_path.exists():
            model = keras.models.load_model(str(model_path))
            models.append(model)
            print(f"  ✅ Loaded {model_dir.name}")
        else:
            print(f"  ⚠️  {model_dir.name} - final_model.h5 not found")
    
    return models

def evaluate_individual_models(models, val_gen):
    """Evaluate each model individually"""
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 80)
    
    results = []
    for i, model in enumerate(models, 1):
        print(f"\nEvaluating Model {i}...")
        result = model.evaluate(val_gen, verbose=0)
        results.append(result)
        
        print(f"  Loss: {result[0]:.4f}")
        print(f"  Accuracy: {result[1]*100:.2f}%")
        if len(result) > 2:
            print(f"  Top-2 Accuracy: {result[2]*100:.2f}%")
        
        # Reset generator
        val_gen.reset()
    
    return results

def evaluate_ensemble(models, val_gen):
    """Evaluate ensemble performance"""
    print("\n" + "=" * 80)
    print("ENSEMBLE PERFORMANCE")
    print("=" * 80)
    
    print(f"\nPredicting with {len(models)}-model ensemble...")
    
    all_predictions = []
    y_true = []
    
    for batch_x, batch_y in val_gen:
        # Get predictions from each model
        batch_preds = []
        for model in models:
            pred = model.predict(batch_x, verbose=0)
            batch_preds.append(pred)
        
        # Average predictions
        avg_pred = np.mean(batch_preds, axis=0)
        all_predictions.append(avg_pred)
        y_true.append(batch_y)
        
        if len(all_predictions) * val_gen.batch_size >= val_gen.samples:
            break
    
    # Concatenate all predictions
    all_predictions = np.vstack(all_predictions)[:val_gen.samples]
    y_true = np.vstack(y_true)[:val_gen.samples]
    
    # Calculate metrics
    y_pred = np.argmax(all_predictions, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    
    accuracy = (y_pred == y_true_labels).mean()
    
    # Top-2 accuracy
    top2_pred = np.argsort(all_predictions, axis=1)[:, -2:]
    top2_accuracy = np.mean([y_true_labels[i] in top2_pred[i] for i in range(len(y_true_labels))])
    
    print(f"\n📊 Ensemble Results:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Top-2 Accuracy: {top2_accuracy*100:.2f}%")
    
    return accuracy, top2_accuracy, y_pred, y_true_labels

def per_class_analysis(y_pred, y_true, class_names):
    """Analyze per-class performance"""
    print("\n" + "=" * 80)
    print("PER-CLASS PERFORMANCE")
    print("=" * 80)
    
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean()
            print(f"  {class_name:<15} Accuracy: {class_acc*100:>6.2f}% ({mask.sum()} samples)")

def calculate_improvement(individual_results, ensemble_accuracy):
    """Calculate improvement from ensemble"""
    individual_accuracies = [r[1] for r in individual_results]
    avg_individual = np.mean(individual_accuracies)
    best_individual = max(individual_accuracies)
    
    print("\n" + "=" * 80)
    print("ENSEMBLE IMPROVEMENT ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 Individual Models:")
    for i, acc in enumerate(individual_accuracies, 1):
        print(f"  Model {i}: {acc*100:.2f}%")
    
    print(f"\n📈 Statistics:")
    print(f"  Average: {avg_individual*100:.2f}%")
    print(f"  Best: {best_individual*100:.2f}%")
    print(f"  Worst: {min(individual_accuracies)*100:.2f}%")
    print(f"  Std Dev: {np.std(individual_accuracies)*100:.2f}%")
    
    print(f"\n🎯 Ensemble:")
    print(f"  Accuracy: {ensemble_accuracy*100:.2f}%")
    
    improvement_vs_avg = (ensemble_accuracy - avg_individual) * 100
    improvement_vs_best = (ensemble_accuracy - best_individual) * 100
    
    print(f"\n✨ Improvement:")
    print(f"  vs Average: +{improvement_vs_avg:.2f}%")
    print(f"  vs Best: +{improvement_vs_best:.2f}%")
    
    if improvement_vs_avg > 2:
        print(f"  ✅ Excellent ensemble gain!")
    elif improvement_vs_avg > 1:
        print(f"  ✅ Good ensemble gain")
    else:
        print(f"  ⚠️  Modest ensemble gain")

if __name__ == '__main__':
    print("=" * 80)
    print("ENSEMBLE EVALUATION")
    print("=" * 80)
    
    # Load models
    models = load_ensemble_models()
    
    if not models:
        print("\n❌ No models found!")
        print("Make sure ensemble_models/tf_files_ensemble/ contains trained models")
        exit(1)
    
    print(f"\n✅ Loaded {len(models)} models successfully")
    
    # Load validation data
    print("\n📊 Loading validation data...")
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Try to find dataset
    dataset_paths = [
        'training_dataset/training_dataset',
        'training_dataset',
        '../training_dataset/training_dataset'
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if Path(path).exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ Dataset not found!")
        exit(1)
    
    val_gen = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"  ✅ Validation samples: {val_gen.samples}")
    
    # Load class names
    class_names = list(val_gen.class_indices.keys())
    
    # Evaluate individual models
    individual_results = evaluate_individual_models(models, val_gen)
    
    # Evaluate ensemble
    ensemble_acc, ensemble_top2, y_pred, y_true = evaluate_ensemble(models, val_gen)
    
    # Per-class analysis
    per_class_analysis(y_pred, y_true, class_names)
    
    # Calculate improvement
    calculate_improvement(individual_results, ensemble_acc)
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 80)
    
    # Summary
    avg_individual = np.mean([r[1] for r in individual_results])
    improvement = (ensemble_acc - avg_individual) * 100
    
    print(f"\n📊 SUMMARY:")
    print(f"  Individual Average: {avg_individual*100:.2f}%")
    print(f"  Ensemble: {ensemble_acc*100:.2f}%")
    print(f"  Improvement: +{improvement:.2f}%")
    
    if ensemble_acc > 0.85:
        print(f"\n✅ EXCELLENT! Ensemble achieved >85% accuracy!")
    elif ensemble_acc > 0.82:
        print(f"\n✅ GOOD! Ensemble improved over single model!")
    else:
        print(f"\n⚠️  Ensemble accuracy is similar to individual models")
