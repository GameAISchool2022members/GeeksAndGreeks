# Readme

## Installation

```
conda create -n aissjam python=3.10
python -m pip install -r requirements.txt
```

## TODOs
- ~~Replace border walls after first enemy movement~~
- Enemy-player collision logic
- Increasing number of spawned enemies as episode progresses (also terminate ep. after max. number of steps)
- Integrate with RL training loop (and train the badboi)
- Ensure all empty tiles are connected
- Allow human player to place/delete walls
- Add sprites to render function