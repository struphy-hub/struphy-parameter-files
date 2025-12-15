# Collection of Struphy simulation parameters

This is a collection of parameter (launch) files for simulations with [Struphy](https://github.com/struphy-hub/struphy).

In order to run a simulation, first [install Struphy](https://github.com/struphy-hub/struphy#quick-install) and then clone this repo. You can then navigate to the folder of the desired simulation, which contains

* a parameter file starting with `params_`
* a post-processing file starting with `pproc_`

Run first the parameter file and then inspect the results by running the post-processing file.


## Categorization

Struphy simulations are categorized by model names first (top-level folders). For each model you can find different simulations (Physics scenarios) in the respective folder. 

There are also top-level folders for specific publications, i.e. for reproducing exactly the simulations in the respective publication.

## Example

Let us run the case of the 1d1v two-stream instability of the model `VlasovAmpereOneSpecies`. Assuming that **Struphy is installed** and you are in the root of this repo, run

```
cd VlasovAmpereOneSpecies
cd two_stream
python params_two_stream.py
python pproc_two_stream.py
```