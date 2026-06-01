# Offboard Radar Controller Refactor

This refactor keeps the executable entry point as:

```bash
python3 offboard_ctrl.py
```

## File structure

```text
offboard_ctrl.py                # Main executable entry point
offboard_ctrl/
├── __init__.py
├── config.py                   # Mission constants and tuning values
├── node.py                     # ROS 2 node and state machine
├── transforms.py               # Radar/body/local frame transforms
└── utils.py                    # Math and message helper functions
```

## How to use in your ROS 2 package

Copy both `offboard_ctrl.py` and the `offboard_ctrl/` folder into your script/source directory.
Then make the main file executable:

```bash
chmod +x offboard_ctrl.py
```

Run it the same way as before:

```bash
python3 offboard_ctrl.py
```

For tuning, edit `offboard_ctrl/config.py` instead of searching through the full node file.
