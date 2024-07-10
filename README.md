# Vector space embeddings and data maps for cyber defense

This repository shows notebooks for experimenting with the computation of vector space embeddings and data maps of telemetry that aim to support cyber defense processes.
All this code was used to generate the results and images shown in the eponymous SciPy 2024 talk
([slides](scipy2024-slides.pdf)).
It is provided here for curious folks to try these experiments by themselves, and perhaps even apply them to their own data.

## Setup

I use [Conda](https://docs.conda.io/projects/conda/en/stable/) to put together the requisite computing environment.
Simply use the included environment file by running

```sh
conda env create
```

If you intend to run this out of the included Jupyter Lab instance, you are good to go.
If instead your workstation consists in a Jupyterhub server, you may need to install the environment as a bespoke kernel:

```sh
conda activate acme3-mapping
python -m ipykernel install --user --name acme3-mapping --display-name "ACME3 data maps"
```

It takes a few seconds for **ACME3 data maps** to show up as a kernel option you can select when starting a new notebook.
From there, when you open any notebook from this repository for the first time, change its kernel to **ACME3 data maps**.

## Notebook index

It is highly recommended to run the notebooks in numerical order, as any may expect to use results computed in a previous one.

0. **[Gather and engineer dataset](0%20Gather%20and%20engineer%20dataset.ipynb)**: this notebook guides you in downloading and filtering the _stdview_ summary of the ACME3 dataset we embed and map.
1. **[Command lines - Bags of words](1%20Command%20lines%20-%20Bags%20of%20words.ipynb)**: we compute a simple bag-of-words embedding of command lines attached to processes, and produce an interactive map of the result.
2. **[Command lines - Wasserstein embedding](2%20Command%20lines%20-%20Wasserstein%20embedding.ipynb)**: we introduce the more sophisticated Wasserstein vector space embedding method, and apply it to command lines, resulting in an improved data map.
3. **[Processes as bags of code images](3%20Processes%20as%20bags%20of%20DLLs.ipynb)**: this takes a different lense on process instances, examining them from patterns of similarity induced by loading similar sets of code images.
4. **[Hosts as bags of processes over time](4%20Hosts%20as%20bags%20of%20processes%20over%20time.ipynb)**: we pivot from comparative process analysis, and uses the process representation to construct and map an embedding of hosts running these processes, with a temporal component.
5. **[Comparing data maps](5%20Comparing%20data%20maps.ipynb)**: this presents a methodology for appraising visually the differences between data maps induced by embedding methods.

## Issues and comments

You may discuss all this publicly by opening an [issue](https://github.com/TutteInstitute/acme3-mapping/issues).
We reserve the right to ask you to submit a PR in support of your arguments. ;-)
