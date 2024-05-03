# eventbee
Eventbee is a library to process and visualize behavior events extracted from [Beepose](https://github.com/jachansantiago/beepose) and [Plotbee](https://github.com/jachansantiago/plotbee).

## Installation

### Requirements

For the kernel running the notebooks/scripts, install with conda/mamba:
```
mamba env create --file requirements.yml -n events
```

If you run jupyterlab from the same environment, install it also:
```
mamba install jupyterlab
```

If jupyterlab server is separate from the kernel environment, make sure to install `ipyml` in server environment
```
mamba install ipympl
```
and that versions are compatible with the kernel. Also add kernel in list of visible kernels
```
python -m ipykernel install --name 'events' --display-name "events" --user
```


### Download and install eventbee

```
# Clone repository
git clone https://github.com/megretlab/cyndibee_event_analysis.git
cd cyndibee_event_analysis
```

Editable version of the eventbee package in your activated environment. 
Use `--no-deps` to avoid pip installing packages. Make sure all dependencies are installed using conda beforehand if conda is used.
```
conda activate events
cd cyndibee_event_analysis
pip install --no-build-isolation --no-deps -e .
```

To uninstall, run
```
pip uninstall eventbee
```

Some notebook using detailled visualization of the videos may also require `plotbee`. Check https://github.com/jachansantiago/plotbee for requirements.
```
# Clone repository
git clone https://github.com/jachansantiago/plotbee.git
cd plotbee
pip install --no-build-isolation --no-deps -e .
# pip uninstall eventbee
```

## Notebooks

- [event_analysis_preparation.ipynb](notebooks/event_analysis_preparation.ipynb)  prepares event dataset from beepose raw files
- [event_analysis_visualization.ipynb](notebooks/event_analysis_visualization.ipynb)  is the main notebook to visualize already prepared event dataset


### Troubleshooting

#### Widgets

In cells with
```
%matplotlib widget
```
matplotlib widget requires `ipympl` installed in kernel environment with a compatible version to jupyterlab server environment.
Check this if slider widgets do not appear, or an error about widgets is returned.
See: https://matplotlib.org/ipympl/installing.html

#### Arrow keys

If arrow keys used in widgets also generate cell change, disable them in the Jupyterlab "Settings/Settings Editor/JSON Settings Editor/User Preferences". For instance:
```
"shortcuts": [
         {
            "args": {},
            "command": "notebook:move-cursor-down",
            "keys": [
                "ArrowDown"
            ],
            "selector": ".jp-Notebook.jp-mod-commandMode:not(.jp-mod-readWrite) :focus"
       ,"disabled": true }
...
``` 
