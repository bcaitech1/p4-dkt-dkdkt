[TOC]

# pstage_04_dkt

## Usage
### train.py

```python
python train.py
# 기본값을 사용하는 방법
```



### Custom Arguments

|      command      |                   default                    |  type  |                         description                          |                             etc                              |
| :---------------: | :------------------------------------------: | :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     **--fes**     |                      []                      |  list  |     적용할 Feature Engineering python file의 이름입니다.     |                            미구현                            |
| **--no_fe_cache** |                    False                     |  bool  |              FE cache 기능을 사용하지 않습니다.              |                            미구현                            |
|    **--json**     |                   "latest"                   | string |       config를 지정한 directory의 파일에서 가져옵니다.       | 기본값은 config/train 폴더에서 최근 modified 된 json 파일을 가져옵니다. |
|   **--exp_cfg**   | "./config/train/export/exported_config.json" | string | train의 config 값을 지정한 directory와 파일명으로 export 합니다. |                                                              |
|  **--no_select**  |                    False                     |  bool  |        CLI json 파일 Select 기능을 사용하지 않습니다.        |                                                              |
|  **--val_name**   |                     None                     | string | dataset split 기능을 사용하지 않고, validation_dataset을 따로 사용합니다. |                  data_dir 에서 파일을 찾음                   |

### JSON 기반 Training

#### basic

가장 기본적인 **python train.py**를 실행하면 다음과 같은 모습입니다.

```bash
Select json file from /opt/ml/p4-dkt-dkdkt/code/config/train (ctrl+c to exit): 

> ↰ to the parent directory (..)
  ┣ /opt/ml/p4-dkt-dkdkt/code/config/train/batch_sample.json
  ┣ /opt/ml/p4-dkt-dkdkt/code/config/train/wandb_sample.json
  ┣ /opt/ml/p4-dkt-dkdkt/code/config/train/sample.json
  ┣ /opt/ml/p4-dkt-dkdkt/code/config/train/export
  ┗ /opt/ml/p4-dkt-dkdkt/code/config/train/new_seed
```

JSON 기반 Training의 기본 값인 Select Mode 이며, 해당 기능을 통해 JSON으로 부터 Config를 가져와 학습시킬 수 있습니다.

아래는 기본값 JSON config 샘플 입니다.

```json
{
  "fes": [],
  "no_fe_cache": null,
  "json": null,
  "exp_cfg": "./config/train/export/exported_config.json",
  "no_select": null,
  "seed": 42,
  "device": "gpu",
  "data_dir": "/opt/ml/input/data/train_dataset",
  "asset_dir": "asset/",
  "file_name": "cv_train_data.csv",
  "val_name": null,
  "model_dir": "models/",
  "model_name": "model.pt",
  "output_dir": "output/",
  "test_file_name": "test_data.csv",
  "max_seq_len": 20,
  "num_workers": 1,
  "pin_mem": true,
  "hidden_dim": 64,
  "n_layers": 2,
  "n_heads": 2,
  "drop_out": 0.2,
  "n_epochs": 20,
  "batch_size": 64,
  "lr": 0.0001,
  "clip_grad": 10,
  "patience": 5,
  "log_steps": 50,
  "model": "lstm",
  "optimizer": "adam",
  "scheduler": "plateau"
}
```

만약 선택 기능을 사용하고 싶지 않으시다면, 

```python
python train.py --no_select 
```

위와 같이 **--no_select** command로, disable 할 수 있으며, 이때, JSON config를 사용하지 않고, argument 들의 기본값을 이용합니다.

```
python train.py --no_select --n_epochs 5 --seed 23
```

이때, 각 argument에 대해 값을 지정해줄 수 있습니다. (**--no_select** command가 없다면 해당 값을 무시하고 JSON Select mode를 사용합니다.)



다음과 같은 방식으로 JSON 파일의 directory를 manual 하게 지정할 수도 있습니다.

```python
train.py --no_select --json ./config/train/sample.json
```



이렇게 설정한 값은 **--exp_cfg** command를 이용해 JSON화 할 수 있습니다.

```python
python train.py --no_select --n_epochs 5 --seed 23 --exp_cfg
```

기본값으로 ./config/train/export/exported_config.json"으로 출력하며, 다음과 같은 방법으로 파일명과 directory를 지정해줄 수 있습니다.

```python
python train.py --no_select --n_epochs 5 --seed 23 --exp_cfg "./config/train/new_seed/seed23epoch5.json"
```

경로가 존재하지 않으면, 폴더를 생성하며, 파일명이 겹칠 경우, 접미사로 "_{n}"를 붙여 출력됩니다.

#### Batch JSON Training

JSON 파일의 argument 설정에  list를 씌움으로, 다양한 config를 연속적으로 Training할 수 있습니다.

다음은 Batched Json 파일의 예시입니다.

```json
{
    "fes": [],
    "no_fe_cache": null,
    "json": null,
    "exp_cfg": "./config/train/export/exported_config.json",
    "no_select": null,
    "seed": [42,52,62],
    "device": "gpu",
    "data_dir": "/opt/ml/input/data/train_dataset",
    "asset_dir": "asset/",
    "file_name": "cv_train_data.csv",
    "val_name": null,
    "model_dir": "models/",
    "model_name": "model.pt",
    "output_dir": "output/",
    "test_file_name": "test_data.csv",
    "max_seq_len": 20,
    "num_workers": 1,
    "pin_mem": true,
    "hidden_dim": [32,64,128],
    "n_layers": 2,
    "n_heads": 2,
    "drop_out": 0.2,
    "n_epochs": 20,
    "batch_size": 64,
    "lr": 0.0001,
    "clip_grad": 10,
    "patience": [5,10,15],
    "log_steps": 50,
    "model": "lstm",
    "optimizer": "adam",
    "scheduler": "plateau"
  }
  
```



이때, 다음과 같은 설정으로 총 3번의 Training을 진행하게 됩니다.

| Train | argument | value |
| ----- | -------- | ----- |
| 0     | seed     | 42    |
| 1     | seed     | 52    |
| 2     | seed     | 62    |

| Train | argument   | value |
| ----- | ---------- | ----- |
| 0     | hidden_dim | 32    |
| 1     | hidden_dim | 64    |
| 2     | hidden_dim | 128   |

| Train | argument | value |
| ----- | -------- | ----- |
| 0     | patience | 5     |
| 1     | patience | 10    |
| 2     | patience | 15    |

나머지 지정하지 않은 단일값(상단의 예시로 **model, lr, max_seq_len**)들은 3번의 Training 모두 같은 값을 사용합니다.



만약, 다음과 같이 List 내부의 값의 크기가 다르다면 다음과 같은 오류를 발생시킵니다.

```json
{
...
    "max_seq_len": 20,
    "num_workers": 1,
    "pin_mem": true,
    "hidden_dim": [32,64,128],
    "n_layers": 2,
    "n_heads": 2,
    "drop_out": 0.2,
    ********* n_epochs : 지정값이 다른 값들과 다르게 4개임! 오류 발생
    "n_epochs": [5,10,15,20],
	*********
    "batch_size": [32,64,128],
    "lr": 0.0001,
    "clip_grad": 10,
    "patience": [5,10,15],
...
  }  
```

```bash
 File "/opt/ml/p4-dkt-dkdkt/code/dkt/utils.py", line 140, in check_batch_available
    raise RuntimeError(f"length of argument {arg_name} doesn't match with other batched arguments. check your json file.") 
RuntimeError: length of argument n_epochs doesn't match with other batched arguments. check your json file.
```

