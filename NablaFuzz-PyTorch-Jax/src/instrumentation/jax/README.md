# Instrumentation 

This folder contains code to perform instrumentation on Jax in order to collect dynamic execution information.

We hook the invocation of `791` Jax APIs, and the API names are listed in `jax-apis.txt`.

The key function is `def hijack()` in `hijack.py`, and you need to set the output database of mongodb in `write_tools.py`.

## Usage:

(1) Copy the folder `instrumentation` to the root directory where Jax is installed. For example, if jax is installed with `virtualenv`, then just copy it to `site-packages/jax/instrumentation/`.

(2) Append these lines to the `site-packages/jax/__init__.py`.

```
from jax.instrumentation.hijack import hijack
hijack()
```

Then we execute the code collected in the first stage to trace various dynamic execution information for each API invocation. The outputs will be stored in the database `jax` of mongodb.