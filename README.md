# ∇Fuzz

This is the artifact of the research paper, "Fuzzing Automatic Differentiation in Deep-Learning Libraries" (in submission). The authors make their best attempt to anonymize the artifacts. Please note that it may still be possible that the authors' identities are revealed by deeply studying this repository.

## About

∇Fuzz is a fully automated API-level fuzzer targeting AD in DL libraries, which utilizes differential testing on different execution scenarios to test both first-order and high-order gradients, and also includes automated filtering strategies to remove false positives caused by numerical instability.

This is the ∇Fuzz's implementation on PyTorch, TensorFlow, JAX, and OneFlow.

## Detected Bugs

Until submission, ∇Fuzz has detected 173 bugs in total for PyTorch, TensorFlow, JAX and OneFlow, with 144 already confirmed by developers.

We provide a list of confirmed bug reports on [PyTorch](https://github.com/NablaFuzz/NablaFuzz/blob/main/PyTorch-confirmed-bugs.csv), [TensorFlow](https://github.com/NablaFuzz/NablaFuzz/blob/main/TensorFlow-confirmed-bugs.csv), [JAX](https://github.com/NablaFuzz/NablaFuzz/blob/main/Jax-confirmed-bugs.csv) and [OneFlow](https://github.com/NablaFuzz/NablaFuzz/blob/main/OneFlow-confirmed-bugs.csv).
> Please kindly note that this might reveal our identity, please think twice before clicking into this.

## Getting Started

### Step 1: Requirements

1. Our testing framework leverages [MongoDB](https://www.mongodb.com/) so you need to [install and run MongoDB](https://docs.mongodb.com/manual/installation/) first.

	- After installing MongoDB and before loading the database, run the command `ulimit -n 640000` to adjust the limit that the system resources a process may use. You can see this [document](https://docs.mongodb.com/manual/reference/ulimit/) for more details.

2. Python version >= 3.9.0 (It must support f-string.)

	- highly recommend to use Python 3.9

3. Check our dependent python libraries in `requirements.txt` and install with:

	```
	pip install -r requirements.txt
	```
	
	- For OneFlow, please use the following command to install:
	
	```shell
	python3 -m pip install --find-links https://release.oneflow.info oneflow==0.7.0
	```

4. JAX install

For JAX, if you want to reproduce our results on version `0.3.14`, you could use the following commands to install jaxlib on CPU via the wheel archive:

```
# Install jaxlib on CPU via the wheel archive
pip install jax[cpu]==0.3.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install the jaxlib 0.3.25 CPU wheel directly
pip install jaxlib==0.3.14 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

For more information about JAX installation, please refer to the [official website](https://jax.readthedocs.io/en/latest/installation.html#installing-jax).

### Step 2: Set up Database

Run the following commands in current directory (`NablaFuzz`) to load the database.

```shell
mongorestore NablaFuzz-PyTorch-Jax/dump
mongorestore NablaFuzz-Oneflow/dump
mongorestore NablaFuzz-TensorFlow/dump
```


### Step 3: Run

Before running, you can set the number of mutants for each API (which is 1000 by default):

```shell
NUM_MUTANT=100
```

Also, you can set the number of APIs you want to test (which is -1 by default, meaning all APIs will be tested)

```shell
NUM_API=100
```

#### PyTorch & JAX

First go into the `NablaFuzz-PyTorch-Jax/src` directory,

```shell
cd NablaFuzz-PyTorch-Jax/src
```

Run the following command to start ∇Fuzz to test PyTorch

```shell
python torch_test.py --num $NUM_MUTANT --max_api $NUM_API
```

The output will be put in `NablaFuzz-PyTorch-Jax/output-ad/torch/union` directory by default. 

To filter out the inconsistent gradients caused by numerical instability, you can run the folloing commands:

```shell
python torch-diff-filter.py --num $NUM_MUTANT --max_api $NUM_API
```

Run the following command to start ∇Fuzz to test JAX

```shell
python jax_test.py
```

The output will be put in `NablaFuzz-PyTorch-Jax/output-ad/jax/union` directory by default. 

To filter out the inconsistent gradients caused by numerical instability, you can run the folloing commands:

```shell
python jax-diff-filter.py
```

#### OneFlow

First go into the `NablaFuzz-Oneflow` directory,

```shell
cd NablaFuzz-Oneflow
```

Run the following commands to start ∇Fuzz to test Oneflow

```shell
python oneflow_test.py --num $NUM_MUTANT --max_api $NUM_API
```

The error code snippets will be put in `NablaFuzz-Oneflow/errors`. 


#### TensorFlow

First go into the `NablaFuzz-TensorFlow/src` directory,

```shell
cd NablaFuzz-TensorFlow/src
```

Run the following commands to start ∇Fuzz to test TensorFlow

```shell
python tf_adtest.py --num $NUM_MUTANT --max_api $NUM_API
python tf_filter.py 
```

The outputs will be put in `NablaFuzz-TensorFlow/expr_outputs` by default. 
