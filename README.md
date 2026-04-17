# Unified Frequency-Based Robustness Pipeline


# Train
Для запуска обучения нужно запустить train.py:
```
cd ufbrp
python train.py
```

Параметры для обучения и датасет указываются в конфиге: `ufbrp/presets/config.yaml`
Результаты обучения сохраняются в `logs_dir`, которая указывается в конфиге.

# Evaluation
Для запуска тестирования атак нужно запустить eval.py:
```
cd ufbrp
python eval.py
```

В `ufbrp/presets/config.yaml` указываются теструемые атаки:
```
attack:
  train:
    type: none
  test:
    -
      type: fgsm
      params:
        eps: 8.0
        alpha: 2.5
        mode: zero
    -
      type: pgd
      params:
        eps: 3.0
        alpha: 2.5
        mode: zero
    -
      type: autoattack
      params:
        eps: 3.0
        mode: zero
```

Результаты тестирования сохраняются в директории `results_dir`, которая указывается в конфиге.
Веса загружаются из директории `logs_dir` по пути `<model>/best_model.pt`. Название `model` указывается в конфиге. 