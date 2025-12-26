# Crypto DQN OPS - Cryptocurrency Trading Timing Optimization

## Описание проекта

Этот проект реализует систему оптимизации времени покупки криптовалюты с использованием Rainbow DQN (Deep Q-Network) - современного алгоритма глубокого обучения с подкреплением. Проект решает задачу оптимальной остановки (Optimal Stopping Problem) в контексте инвестирования в криптовалюты по стратегии Dollar Cost Averaging (DCA).

### Основная задача

Агент обучается определять оптимальный момент для покупки криптовалюты в рамках заданного инвестиционного цикла, анализируя исторические данные о ценах и текущее состояние рынка.

### Ключевые особенности

- **Rainbow DQN**: Комбинация нескольких улучшений DQN:

  - Double DQN - уменьшение переоценки Q-значений
  - Dueling Networks - разделение оценки состояния и преимущества действий
  - Prioritized Experience Replay (PER) - приоритизированное воспроизведение опыта
  - Categorical DQN (C51) - распределительное представление Q-значений
  - NoisyNet - параметрический шум для исследования
  - N-step Learning - многошаговое обучение

- **MLOps практики**:
  - Управление конфигурациями через Hydra
  - Логирование экспериментов в MLflow
  - Версионирование данных через DVC
  - Автоматизация проверки кода через pre-commit
  - Поддержка PyTorch Lightning для масштабируемого обучения

### Архитектура

```
crypto_dqn_ops/
├── agents/          # RL агенты
├── models/          # Нейронные сети
├── environment/     # Торговое окружение
├── data/            # Загрузка и обработка данных
├── training/        # PyTorch Lightning модули
├── inference/       # Инференс и предсказания
└── utils/           # Вспомогательные функции
```

## Setup

### Требования

- Python 3.9+
- uv (для управления зависимостями)
- Git
- DVC (опционально, для работы с данными)

### Установка

1. **Клонирование репозитория**

```bash
git clone https://github.com/your-username/crypto-dqn-ops.git
cd crypto-dqn-ops
```

2. **Создание виртуального окружения и установка зависимостей**

```bash
# Установка uv (если еще не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Установка зависимостей
uv sync

# Активация виртуального окружения
source .venv/bin/activate  # на macOS/Linux
# или
.venv\Scripts\activate  # на Windows
```

3. **Настройка pre-commit хуков**

```bash
uv run pre-commit install
```

4. **Инициализация DVC (опционально)**

```bash
dvc init
dvc remote add -d local /tmp/dvc-storage
```

### Проверка установки

Запустите pre-commit на всех файлах:

```bash
uv run pre-commit run -a
```

Все проверки должны пройти успешно (зеленые результаты).

## Train

### Подготовка данных

Данные автоматически загружаются через DVC при первом запуске обучения. Если DVC не настроен, поместите файл `crypto_data.pkl` в директорию `data/`.

### Запуск обучения

**Базовое обучение на Bitcoin (с Lightning Trainer):**

```bash
uv run python commands.py train
```

**Обучение без Lightning (классический цикл):**

```bash
uv run python commands.py train --overrides training.use_lightning=false
```

**Обучение на Ethereum:**

```bash
uv run python commands.py train --overrides data=eth
```

**Обучение с кастомными параметрами:**

```bash
uv run python commands.py train --overrides \
  training.num_frames=500000 \
  training.batch_size=256 \
  training.learning_rate=1e-4 \
  data.window_size=60
```

### Параметры обучения

Основные параметры настраиваются через Hydra конфигурации в `configs/`:

- `configs/config.yaml` - основная конфигурация
- `configs/data/` - настройки данных (BTC/ETH)
- `configs/model/` - параметры модели
- `configs/training/` - гиперпараметры обучения

### Мониторинг обучения

Обучение логируется в MLflow. Для просмотра метрик:

```bash
# Запуск MLflow UI
uv run mlflow ui --port 8080

# Откройте в браузере
http://127.0.0.1:8080
```

В MLflow доступны:

- Графики loss и mean_score
- Гиперпараметры эксперимента
- Git commit ID
- Сохраненные модели

## Production Preparation

### Экспорт в ONNX

Для продакшена модель конвертируется в ONNX формат:

```bash
uv run python commands.py export_onnx \
  --model_path trained_models/BTC_777/model_final.pth \
  --output_path trained_models/model.onnx
```

ONNX модель:

- Независима от PyTorch
- Оптимизирована для инференса
- Поддерживает различные runtime (ONNX Runtime, TensorRT)

### Экспорт в TensorRT

Для максимальной производительности на NVIDIA GPU используйте TensorRT:

```bash
# Создайте скрипт export_tensorrt.sh
bash scripts/export_tensorrt.sh trained_models/model.onnx trained_models/model.trt
```

Содержимое `scripts/export_tensorrt.sh`:

```bash
#!/bin/bash
ONNX_MODEL=$1
TRT_MODEL=$2

trtexec \
  --onnx=$ONNX_MODEL \
  --saveEngine=$TRT_MODEL \
  --explicitBatch \
  --fp16 \
  --workspace=4096
```

### Комплектация поставки

Для развертывания модели необходимы:

1. **Файлы модели:**

   - `model_final.pth` или `model.onnx` или `model.trt`

2. **Конфигурация:**

   - `configs/config.yaml` (параметры модели)

3. **Код инференса:**

   - `crypto_dqn_ops/inference/predictor.py`
   - `crypto_dqn_ops/models/rainbow_network.py`
   - `crypto_dqn_ops/utils/helpers.py`

4. **Зависимости (минимальные для инференса):**
   ```
   torch>=2.0.0
   numpy>=1.24.0
   ```

## Infer

### Вариант 1: Скриптовый инференс

**Инференс на тестовых данных:**

```bash
uv run python commands.py infer \
  --model_path trained_models/BTC_777/model_final.pth \
  --data_path data/crypto_data.pkl \
  --crypto BTC
```

**Инференс на Ethereum:**

```bash
uv run python commands.py infer \
  --model_path trained_models/ETH_777/model_final.pth \
  --data_path data/crypto_data.pkl \
  --crypto ETH
```

### Вариант 2: Inference Server (FastAPI)

**Запуск сервера:**

```bash
uv run python commands.py serve
```

Или на другом порту:

```bash
uv run python commands.py serve --port 8000
```

**Использование API:**

```bash
# Health check
curl http://127.0.0.1:5000/health

# Информация о модели
curl http://127.0.0.1:5000/model/info

# Предсказание
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [[0.5, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                      0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
                      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                      0.5, 0.3]]
  }'
```

**API Documentation:**

Откройте в браузере: `http://127.0.0.1:5000/docs`

**Endpoints:**

- `GET /` - статус сервера
- `GET /health` - health check
- `GET /model/info` - информация о модели
- `POST /predict` - предсказания

### Формат входных данных

Модель принимает наблюдения размерности `(window_size + 2)`:

- `position_value` (1 значение) - нормализованная позиция текущей цены
- `remaining_time` (1 значение) - оставшееся время в цикле (0-1)
- `price_history` (window_size значений) - нормализованная история цен

**Пример данных:**

```python
import numpy as np
from crypto_dqn_ops.inference.predictor import CryptoPredictor

# Загрузка модели
predictor = CryptoPredictor(
    model_path="trained_models/model_final.pth",
    obs_dim=32,  # window_size=30 + 2
)

# Пример наблюдения
observation = np.array([
    0.5,  # position_value
    0.8,  # remaining_time
    # 30 значений нормализованных цен
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
    0.6, 0.65, 0.7, 0.75, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,
    0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05
])

# Предсказание действия
action = predictor.predict(observation)
# action = 0 (держать) или 1 (покупать)
```

### Формат выходных данных

Модель возвращает:

- **action** (int): 0 - держать, 1 - покупать
- **q_values** (array): Q-значения для каждого действия

## Структура проекта

```
crypto-dqn-ops/
├── crypto_dqn_ops/          # Основной пакет
│   ├── agents/              # RL агенты
│   ├── models/              # Нейронные сети
│   ├── environment/         # Торговое окружение
│   ├── data/                # Загрузка данных
│   ├── training/            # PyTorch Lightning
│   ├── inference/           # Инференс
│   └── utils/               # Утилиты
├── configs/                 # Hydra конфигурации
│   ├── config.yaml
│   ├── data/
│   ├── model/
│   └── training/
├── data/                    # Данные (управляются DVC)
├── trained_models/          # Обученные модели
├── plots/                   # Графики и визуализации
├── scripts/                 # Вспомогательные скрипты
├── commands.py              # CLI интерфейс
├── pyproject.toml           # Poetry конфигурация
├── .pre-commit-config.yaml  # Pre-commit хуки
├── .dvc/                    # DVC конфигурация
└── README.md
```

## Разработка

### Code Quality

Проект использует следующие инструменты:

- **black** - форматирование кода
- **isort** - сортировка импортов
- **flake8** - линтинг
- **prettier** - форматирование YAML/JSON/Markdown

Запуск всех проверок:

```bash
uv run pre-commit run -a
```

### Тестирование

```bash
uv run pytest tests/
```

### Добавление новых зависимостей

```bash
uv add package-name
```

## Метрики и результаты

Модель оценивается по следующим метрикам:

1. **Mean Score** - средний reward за последние 10 эпизодов
2. **Loss** - функция потерь DQN
3. **Сравнение со стратегиями:**
   - Покупка в первый день цикла
   - Покупка в последний день цикла
   - Покупка в случайный день
   - Покупка по средней цене

Результаты сохраняются в `plots/` и логируются в MLflow.

## Лицензия

MIT License

## Контакты

Для вопросов и предложений создавайте issues в репозитории.
