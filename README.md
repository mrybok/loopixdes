📧 [Loopix](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/piotrowska)DES
---
The Loopix Anonymity System Discrete-Event Simulator

---

Requirements:
- `gym`
- `tqdm`
- `numpy`
- `simpy`
- `tensorflow`

*Tensorflow 2 is required.

---

To run:

`$python runner.py` 

or manually:

```
from loopixdes.env import LoopixEnv
from loopixdes.defaults import DEFAULT_PARAMS

env = LoopixEnv()
state = env.reset(seed=0, options={})

for _ in range(10):
    params = np.array(list(DEFAULT_PARAMS.values()))
    state, reward, done, _ = env.step(params) 
```
---
For Tensorboard:

`$tensorboard --logdir runs`

---
Full documentation coming soon...