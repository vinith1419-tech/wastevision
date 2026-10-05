# %% [markdown]
# # WasteVision Phase 6 — controlled YOLOv8n improvement (Google Colab)
#
# Run top-to-bottom on a Colab T4 runtime. Upload/copy the complete project to
# Drive first and set `PROJECT_ROOT` below. This notebook never trains on test,
# never edits `models/best.pt`, and does not promote a checkpoint automatically.
# Candidate selection is validation-only. Test/hard-test evaluation happens
# once after selection. Keep the resulting output directory for reproducibility.

# %%
from pathlib import Path
import json, math, os, random, shutil, sys, time
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# In Colab, mount Drive in a separate cell if it is not already mounted.
from google.colab import drive
drive.mount('/content/drive')

# Change this if the copied project folder has a different name.
PROJECT_ROOT = Path('/content/drive/MyDrive/Waste detection')
DATASET_ROOT = PROJECT_ROOT / 'dataset'
BASELINE_WEIGHTS = PROJECT_ROOT / 'models' / 'best.pt'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'phase6'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
assert (DATASET_ROOT / 'data.yaml').is_file(), f'Missing {DATASET_ROOT / "data.yaml"}'
assert BASELINE_WEIGHTS.is_file(), f'Missing official baseline {BASELINE_WEIGHTS}'

# %% [markdown]
# ## Install and record runtime
#
# Colab's preinstalled torch/CUDA stack is retained. Ultralytics is installed
# without replacing torch; restart the runtime only if Colab explicitly asks.

# %%
import subprocess
try:
    import ultralytics  # noqa: F401
    import yaml  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'pyyaml'], check=True)

# %%
import torch, yaml, ultralytics
from ultralytics import YOLO
RUNTIME_INFO = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'ultralytics': ultralytics.__version__,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(RUNTIME_INFO)
assert torch.cuda.is_available(), 'Select a Colab GPU runtime before training.'

# %% [markdown]
# ## Audit the exact copied dataset before any training
#
# The audit is read-only with respect to images/labels. Stop if its class order,
# geometry, or split pairing check fails. Keep the JSON and Markdown outputs.

# %%
import subprocess
audit_script = PROJECT_ROOT / 'scripts' / 'phase6_dataset_audit.py'
assert audit_script.is_file(), 'Copy the updated project, including scripts/phase6_dataset_audit.py, to Drive.'
subprocess.run([sys.executable, str(audit_script)], cwd=PROJECT_ROOT, check=True)
audit = json.loads((PROJECT_ROOT / 'docs' / 'phase6_dataset_audit.json').read_text())
assert audit['data_yaml']['class_order_valid'] and not audit['errors'], audit['errors']
expected_images = {'train': 477, 'val': 96, 'test': 64}
expected_objects = {'plastic': 893, 'glass': 53, 'paper': 87, 'metal': 302, 'cardboard': 23}
assert {k: v['images'] for k, v in audit['splits'].items()} == expected_images
assert {k: v['total_objects'] for k, v in audit['class_distribution'].items()} == expected_objects
print('Dataset audited. Test split remains locked until candidate selection.')

# %% [markdown]
# ## Prepare immutable dataset YAMLs and capped training-only oversampling
#
# No validation or test file is copied into training. The balancing variant
# repeats only minority-containing training images, with a per-source cap of
# three total copies (original + at most two repeats). Standard online YOLO
# transformations then vary those examples; no synthetic labels are invented.

# %%
CLASS_NAMES = ['plastic', 'glass', 'paper', 'metal', 'cardboard']
BASE_YAML = yaml.safe_load((DATASET_ROOT / 'data.yaml').read_text())
assert BASE_YAML['names'] == CLASS_NAMES and BASE_YAML['nc'] == 5

def absolute_yaml(path: Path, train_path: Path) -> Path:
    cfg = {'path': str(DATASET_ROOT), 'train': str(train_path),
           'val': str(DATASET_ROOT / 'images' / 'val'),
           'test': str(DATASET_ROOT / 'images' / 'test'),
           'nc': 5, 'names': CLASS_NAMES}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path

ORIGINAL_YAML = absolute_yaml(OUTPUT_ROOT / 'data_original.yaml', DATASET_ROOT / 'images' / 'train')
BALANCED_ROOT = OUTPUT_ROOT / 'balanced_train_only'
BALANCED_IMAGES = BALANCED_ROOT / 'images' / 'train'
BALANCED_LABELS = BALANCED_ROOT / 'labels' / 'train'
if BALANCED_ROOT.exists():
    shutil.rmtree(BALANCED_ROOT)
BALANCED_IMAGES.mkdir(parents=True)
BALANCED_LABELS.mkdir(parents=True)

# Class priority and total source-image copies. The hard cap avoids exploding
# the dataset, while prioritizing cardboard, glass, then paper.
TOTAL_COPIES = {'cardboard': 3, 'glass': 2, 'paper': 2, 'metal': 1, 'plastic': 1}
train_images = sorted(p for p in (DATASET_ROOT / 'images' / 'train').iterdir() if p.is_file())
manifest = []
for image in train_images:
    label = DATASET_ROOT / 'labels' / 'train' / f'{image.stem}.txt'
    assert label.is_file(), f'Missing training label: {label}'
    ids = set()
    for line in label.read_text().splitlines():
        fields = line.split()
        if fields:
            ids.add(int(fields[0]))
    names = [CLASS_NAMES[i] for i in ids]
    copies = max([TOTAL_COPIES[n] for n in names] or [1])
    for copy_id in range(copies):
        suffix = '' if copy_id == 0 else f'__repeat{copy_id}'
        target_stem = image.stem + suffix
        shutil.copy2(image, BALANCED_IMAGES / f'{target_stem}{image.suffix.lower()}')
        shutil.copy2(label, BALANCED_LABELS / f'{target_stem}.txt')
        manifest.append({'generated_image': target_stem + image.suffix.lower(),
                         'source_image': str(image.relative_to(PROJECT_ROOT)),
                         'source_label': str(label.relative_to(PROJECT_ROOT)),
                         'source_split': 'train', 'classes_in_source': names,
                         'copy_index': copy_id, 'total_copies': copies})
manifest_path = OUTPUT_ROOT / 'oversampling_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2))
BALANCED_YAML = absolute_yaml(OUTPUT_ROOT / 'data_balanced.yaml', BALANCED_IMAGES)
print('Original train images:', len(train_images), 'balanced train image files:', len(manifest))
print('Only files in dataset/images/train were eligible for repetition.')

# %% [markdown]
# ## Fixed controlled experiment matrix
#
# All experiments start from the same COCO-pretrained `yolov8n.pt`, fixed seed,
# and 640 image size. A uses the original training set and mild, visually
# conservative augmentations. B uses the capped training-only oversampling and
# standard Ultralytics augmentation. C uses the same balanced data with a
# carefully bounded geometric/color policy. MixUp/copy-paste/vertical flip are
# disabled to avoid synthetic material cues or unsafe box compositing.

# %%
EXPERIMENTS = [
    {'name': 'A_original_mild_aug', 'yaml': ORIGINAL_YAML, 'epochs': 120,
     'patience': 25, 'degrees': 5.0, 'translate': 0.10, 'scale': 0.40,
     'fliplr': 0.50, 'flipud': 0.0, 'hsv_h': 0.01, 'hsv_s': 0.30,
     'hsv_v': 0.20, 'mosaic': 0.70, 'mixup': 0.0, 'copy_paste': 0.0},
    {'name': 'B_capped_balance_default_aug', 'yaml': BALANCED_YAML, 'epochs': 180,
     'patience': 30},
    {'name': 'C_capped_balance_tuned_aug', 'yaml': BALANCED_YAML, 'epochs': 180,
     'patience': 30, 'degrees': 5.0, 'translate': 0.10, 'scale': 0.50,
     'fliplr': 0.50, 'flipud': 0.0, 'hsv_h': 0.01, 'hsv_s': 0.30,
     'hsv_v': 0.20, 'mosaic': 0.50, 'mixup': 0.0, 'copy_paste': 0.0},
]
COMMON = {'imgsz': 640, 'batch': -1, 'seed': SEED, 'deterministic': True,
          'pretrained': True, 'device': 0, 'workers': 4, 'plots': True,
          'save': True, 'cache': False, 'close_mosaic': 10, 'amp': True,
          'verbose': True}
training_configs = []
for exp in EXPERIMENTS:
    args = {**COMMON, **{k: v for k, v in exp.items() if k not in ('name', 'yaml')}}
    model = YOLO('yolov8n.pt')
    started = time.time()
    model.train(data=str(exp['yaml']), epochs=exp['epochs'], patience=exp['patience'],
                project=str(OUTPUT_ROOT / 'runs'), name=exp['name'], exist_ok=True,
                **{k: v for k, v in args.items() if k not in ('epochs', 'patience')})
    run_dir = OUTPUT_ROOT / 'runs' / exp['name']
    best = run_dir / 'weights' / 'best.pt'
    assert best.is_file(), f'No best checkpoint produced for {exp["name"]}'
    training_configs.append({'experiment': exp['name'], 'yaml': str(exp['yaml']),
                             'requested_epochs': exp['epochs'], 'patience': exp['patience'],
                             'elapsed_seconds': time.time() - started,
                             'best_checkpoint': str(best), 'args': args})
    (run_dir / 'phase6_config.json').write_text(json.dumps(training_configs[-1], indent=2))
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## Validation-only checkpoint selection
#
# Each best checkpoint is evaluated on validation with plots. Rank primarily
# by mAP@0.5:0.95; use mAP@0.5, recall and minority-class AP as tie-breakers.
# No test metric is read in this selection cell.

# %%
def scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def metrics_record(metrics):
    box = metrics.box
    names = metrics.names
    ap = np.asarray(getattr(box, 'all_ap', []))
    ap_ids = list(getattr(box, 'ap_class_index', []))
    per_class = {}
    for row, class_id in zip(ap, ap_ids):
        per_class[names[int(class_id)]] = {'ap50': float(row[0]), 'map50_95': float(np.mean(row))}
    precision = float(np.mean(box.p)) if len(box.p) else 0.0
    recall = float(np.mean(box.r)) if len(box.r) else 0.0
    return {'precision': precision, 'recall': recall,
            'f1_derived_from_aggregate_p_r': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'map50': float(box.map50), 'map50_95': float(box.map), 'per_class': per_class}

validation = {}
for item in training_configs:
    name = item['experiment']
    model = YOLO(item['best_checkpoint'])
    metrics = model.val(data=str(item['yaml']), split='val', imgsz=640, device=0,
                        plots=True, project=str(OUTPUT_ROOT / 'validation'),
                        name=name, exist_ok=True, verbose=True)
    validation[name] = metrics_record(metrics)
    (OUTPUT_ROOT / f'{name}_validation.json').write_text(json.dumps(validation[name], indent=2))

# Validate the official baseline on the same validation split so that the
# validation improvement gate is a same-runtime, same-data comparison.
baseline_model = YOLO(str(BASELINE_WEIGHTS))
baseline_val_obj = baseline_model.val(data=str(ORIGINAL_YAML), split='val', imgsz=640,
    device=0, plots=True, project=str(OUTPUT_ROOT / 'validation'), name='baseline',
    exist_ok=True, verbose=True)
baseline_validation = metrics_record(baseline_val_obj)
(OUTPUT_ROOT / 'baseline_validation.json').write_text(json.dumps(baseline_validation, indent=2))
del baseline_model
torch.cuda.empty_cache()

def rank_key(name):
    m = validation[name]
    minority = m['per_class']
    weak_ap = [minority[n]['map50_95'] for n in ('glass', 'paper', 'cardboard') if n in minority]
    return (m['map50_95'], m['map50'], m['recall'], sum(weak_ap) / len(weak_ap) if weak_ap else -1)

selected_name = max(validation, key=rank_key)
print('Validation-only selected candidate:', selected_name, validation[selected_name])
(OUTPUT_ROOT / 'validation_selection.json').write_text(json.dumps(
    {'selection_basis': 'validation only: mAP50-95, then mAP50, recall, weak-class mean AP',
     'selected': selected_name, 'metrics': validation}, indent=2))

# %% [markdown]
# ## Final held-out test evaluation (only after selection)
#
# Evaluate the selected checkpoint and the official baseline once, on the same
# test split. These outputs must not be used to return to tuning/selection.
# This also emits confusion matrices, PR/P/R/F1 curves and per-class AP data.

# %%
selected_weights = next(x['best_checkpoint'] for x in training_configs if x['experiment'] == selected_name)
test_results = {}
for label, weights in [('baseline', BASELINE_WEIGHTS), ('candidate', Path(selected_weights))]:
    model = YOLO(str(weights))
    metrics = model.val(data=str(ORIGINAL_YAML), split='test', imgsz=640, device=0,
                        plots=True, project=str(OUTPUT_ROOT / 'final_test'),
                        name=label, exist_ok=True, verbose=True)
    test_results[label] = metrics_record(metrics)
    (OUTPUT_ROOT / f'{label}_test_metrics.json').write_text(json.dumps(test_results[label], indent=2))
    del model
    torch.cuda.empty_cache()

comparison_images = sorted((DATASET_ROOT / 'images' / 'test').glob('*'))
for label, weights in [('baseline', BASELINE_WEIGHTS), ('candidate', Path(selected_weights))]:
    model = YOLO(str(weights))
    model.predict(source=[str(p) for p in comparison_images], imgsz=640, device=0,
                  conf=0.001, save=True, project=str(OUTPUT_ROOT / 'runs'),
                  name=f'test_predictions_{label}', exist_ok=True, verbose=False)
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## Optional independent hard-test evaluation
#
# This runs only if labeled hard-test images have been independently supplied.
# Never use these metrics to choose augmentation, thresholds, or checkpoints.

# %%
HARD_IMAGES = PROJECT_ROOT / 'dataset' / 'hard_test' / 'images'
HARD_LABELS = PROJECT_ROOT / 'dataset' / 'hard_test' / 'labels'
hard_results = {'status': 'not_run', 'reason': 'No labeled hard-test images supplied.'}
hard_files = [p for p in HARD_IMAGES.glob('*') if p.is_file()] if HARD_IMAGES.is_dir() else []
hard_labels = [p for p in HARD_LABELS.glob('*.txt') if p.is_file()] if HARD_LABELS.is_dir() else []
if hard_files and hard_labels and len(hard_files) == len(hard_labels):
    image_stems = {p.stem for p in hard_files}
    label_stems = {p.stem for p in hard_labels}
    assert image_stems == label_stems, 'Every hard-test image needs its same-stem YOLO label.'
    hard_yaml = OUTPUT_ROOT / 'data_hard_test.yaml'
    hard_yaml.write_text(yaml.safe_dump({'path': str(PROJECT_ROOT / 'dataset' / 'hard_test'),
        'train': str(HARD_IMAGES), 'val': str(HARD_IMAGES), 'test': str(HARD_IMAGES),
        'nc': 5, 'names': CLASS_NAMES}, sort_keys=False))
    hard_results = {}
    for label, weights in [('baseline', BASELINE_WEIGHTS), ('candidate', Path(selected_weights))]:
        model = YOLO(str(weights))
        hard_metrics = model.val(data=str(hard_yaml), split='val', imgsz=640, device=0,
                                 plots=True, project=str(OUTPUT_ROOT / 'hard_test'),
                                 name=label, exist_ok=True, verbose=True)
        hard_results[label] = metrics_record(hard_metrics)
        del model
        torch.cuda.empty_cache()
    # Fixed app threshold used only for object-level hard-scene accounting.
    # Aggregate PR/AP curves above still report performance across confidence.
    from PIL import Image
    import csv
    HARD_CONF = 0.25
    def box_iou(a, b):
        ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0
    def hard_scene_rows(weights, label):
        model = YOLO(str(weights))
        rows = []
        for image_path in sorted(hard_files):
            with Image.open(image_path) as im:
                width, height = im.size
            truth = []
            for line in (HARD_LABELS / f'{image_path.stem}.txt').read_text().splitlines():
                fields = line.split()
                if not fields:
                    continue
                cid, xc, yc, bw, bh = int(fields[0]), *(float(v) for v in fields[1:])
                truth.append({'class_id': cid, 'box': [(xc-bw/2)*width, (yc-bh/2)*height,
                                                       (xc+bw/2)*width, (yc+bh/2)*height]})
            result = model.predict(source=str(image_path), imgsz=640, device=0, conf=HARD_CONF,
                                   save=True, project=str(OUTPUT_ROOT / 'hard_test_overlays'),
                                   name=label, exist_ok=True, verbose=False)[0]
            predicted = []
            if result.boxes is not None:
                for coords, score, cid in zip(result.boxes.xyxy.cpu().numpy(),
                                              result.boxes.conf.cpu().numpy(),
                                              result.boxes.cls.cpu().numpy()):
                    predicted.append({'class_id': int(cid), 'confidence': float(score),
                                      'box': [float(v) for v in coords[:4]]})
            pairs = sorted(((box_iou(t['box'], p['box']), ti, pi)
                            for ti, t in enumerate(truth) for pi, p in enumerate(predicted)), reverse=True)
            used_t, used_p, correct = set(), set(), []
            for overlap, ti, pi in pairs:
                if overlap >= 0.5 and ti not in used_t and pi not in used_p and truth[ti]['class_id'] == predicted[pi]['class_id']:
                    used_t.add(ti); used_p.add(pi); correct.append((ti, pi, overlap))
            confusions = []
            for overlap, ti, pi in pairs:
                if overlap >= 0.5 and ti not in used_t and pi not in used_p:
                    if truth[ti]['class_id'] != predicted[pi]['class_id']:
                        confusions.append({'expected': CLASS_NAMES[truth[ti]['class_id']],
                                           'detected': CLASS_NAMES[predicted[pi]['class_id']],
                                           'confidence': predicted[pi]['confidence'], 'iou': overlap})
                        used_t.add(ti); used_p.add(pi)
            false_negatives = [CLASS_NAMES[truth[i]['class_id']] for i in range(len(truth)) if i not in used_t]
            false_positives = [{'class': CLASS_NAMES[predicted[i]['class_id']],
                                'confidence': predicted[i]['confidence']}
                               for i in range(len(predicted)) if i not in used_p]
            rows.append({'model': label, 'scene_id': image_path.stem, 'image': image_path.name,
                         'expected_objects': len(truth), 'detected_objects': len(predicted),
                         'true_positive_correct_class': len(correct), 'false_positive_count': len(false_positives),
                         'false_positive_details': false_positives, 'false_negative_count': len(false_negatives),
                         'false_negative_classes': false_negatives, 'class_confusions': confusions,
                         'matched_confidences': [predicted[pi]['confidence'] for _, pi, _ in correct],
                         'confidence_threshold': HARD_CONF, 'match_iou_threshold': 0.5})
        del model
        return rows
    hard_detail = {}
    for label, weights in [('baseline', BASELINE_WEIGHTS), ('candidate', Path(selected_weights))]:
        hard_detail[label] = hard_scene_rows(weights, label)
        detail_path = OUTPUT_ROOT / f'hard_test_per_scene_{label}.json'
        detail_path.write_text(json.dumps(hard_detail[label], indent=2))
        with (OUTPUT_ROOT / f'hard_test_per_scene_{label}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(hard_detail[label][0].keys()) if hard_detail[label] else ['scene_id'])
            writer.writeheader(); writer.writerows(hard_detail[label])
(OUTPUT_ROOT / 'hard_test_metrics.json').write_text(json.dumps(hard_results, indent=2))
print('Hard-test result status:', hard_results.get('status', 'measured'))

# %% [markdown]
# ## Comparison table, promotion guard, and package
#
# A model is not automatically promoted. Review all validation/test metrics,
# minority classes, confusion matrices, hard scenes, and false-positive/fn
# overlays. Candidate must improve validation and test localization, retain
# recall, improve weak-class evidence, and improve independent hard-test results.
# A separate human review flag confirms no new serious failure mode. The
# notebook never writes over `models/best.pt`.

# %%
def row(label, m):
    return {'model': label, 'precision': m['precision'], 'recall': m['recall'],
            'f1': m['f1_derived_from_aggregate_p_r'], 'map50': m['map50'], 'map50_95': m['map50_95']}

table = [row('Baseline YOLOv8n 50e (supplied historic test)', {
    'precision': .2331, 'recall': .2423, 'f1_derived_from_aggregate_p_r': .2376,
    'map50': .2111, 'map50_95': .1419}),
    row('Phase 6 candidate', test_results['candidate'])]
for name, m in validation.items():
    table.append(row(f'Phase 6 {name} (validation)', m))
print(json.dumps(table, indent=2))

WEAK_CLASSES = ('glass', 'paper', 'cardboard')
def weak_mean(metrics):
    values = [metrics['per_class'][n]['map50_95'] for n in WEAK_CLASSES if n in metrics['per_class']]
    return float(np.mean(values)) if values else None

base_val, candidate_val = baseline_validation, validation[selected_name]
base_test, candidate_test = test_results['baseline'], test_results['candidate']
gates = {
    'validation_map50_95_gain_at_least_0_01': candidate_val['map50_95'] >= base_val['map50_95'] + 0.01,
    'validation_map50_not_down_more_than_0_01': candidate_val['map50'] >= base_val['map50'] - 0.01,
    'validation_recall_drop_no_more_than_0_03': candidate_val['recall'] >= base_val['recall'] - 0.03,
    'validation_weak_class_mean_ap_improves': weak_mean(candidate_val) is not None and weak_mean(base_val) is not None and weak_mean(candidate_val) >= weak_mean(base_val) + 0.01,
    'test_map50_95_gain_at_least_0_01': candidate_test['map50_95'] >= base_test['map50_95'] + 0.01,
    'test_map50_not_down_more_than_0_01': candidate_test['map50'] >= base_test['map50'] - 0.01,
    'test_recall_drop_no_more_than_0_03': candidate_test['recall'] >= base_test['recall'] - 0.03,
    'test_weak_class_mean_ap_improves': weak_mean(candidate_test) is not None and weak_mean(base_test) is not None and weak_mean(candidate_test) >= weak_mean(base_test) + 0.01,
    'hard_test_measured': hard_results.get('status') != 'not_run',
}
if gates['hard_test_measured']:
    base_hard, candidate_hard = hard_results['baseline'], hard_results['candidate']
    gates.update({
        'hard_test_map50_95_gain_at_least_0_01': candidate_hard['map50_95'] >= base_hard['map50_95'] + 0.01,
        'hard_test_map50_not_down_more_than_0_01': candidate_hard['map50'] >= base_hard['map50'] - 0.01,
        'hard_test_recall_drop_no_more_than_0_03': candidate_hard['recall'] >= base_hard['recall'] - 0.03,
        'hard_test_weak_class_mean_ap_improves': weak_mean(candidate_hard) is not None and weak_mean(base_hard) is not None and weak_mean(candidate_hard) >= weak_mean(base_hard) + 0.01,
    })
else:
    for key in ('hard_test_map50_95_gain_at_least_0_01', 'hard_test_map50_not_down_more_than_0_01',
                'hard_test_recall_drop_no_more_than_0_03', 'hard_test_weak_class_mean_ap_improves'):
        gates[key] = False
MANUAL_ERROR_REVIEW_PASS = False  # Set True only after reviewing comparison overlays and plots.
gates['manual_error_review_pass'] = MANUAL_ERROR_REVIEW_PASS
all_gates_pass = all(gates.values())
(OUTPUT_ROOT / 'promotion_gates.json').write_text(json.dumps(gates, indent=2))
print('Promotion gates:', json.dumps(gates, indent=2))

PROMOTE_AFTER_MANUAL_REVIEW = all_gates_pass
if PROMOTE_AFTER_MANUAL_REVIEW:
    destination = PROJECT_ROOT / 'models' / 'phase6_best.pt'
    shutil.copy2(selected_weights, destination)
    print('Saved candidate separately:', destination)
else:
    print('At least one objective/manual gate failed. No Phase 6 model was promoted; models/best.pt is unchanged.')

package_summary = {'seed': SEED, 'image_size': 640, 'runtime': RUNTIME_INFO,
                   'experiments': training_configs,
                   'validation': validation, 'baseline_validation': baseline_validation,
                   'selected_by_validation': selected_name,
                   'final_test': test_results, 'hard_test': hard_results, 'promotion_gates': gates,
                   'baseline_checkpoint': str(BASELINE_WEIGHTS), 'candidate_checkpoint': str(selected_weights),
                   'class_names': CLASS_NAMES, 'promotion_enabled': PROMOTE_AFTER_MANUAL_REVIEW}
(OUTPUT_ROOT / 'phase6_results.json').write_text(json.dumps(package_summary, indent=2))
print('All Phase 6 run artifacts:', OUTPUT_ROOT)
