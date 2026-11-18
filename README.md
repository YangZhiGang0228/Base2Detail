# Base2Detail

Base2Detail fits a two-stage neural representation to each triangle mesh independently. Stage 1 learns geometry-dependent spectral subdivision from multiresolution supervision. Stage 2 fits scalar displacements along the fixed terminal mesh normals using encoded distances to eight anchors.

## Environment

Python 3.10, PyTorch 2.7.1+cu128, CUDA 12.8, PyTorch3D 0.7.9, NumPy 1.26.4, and SciPy 1.15.3.

## Data preparation

The example provides a reference OBJ, a coarse mesh, and three SSP supervision levels.

```text
data_meshes/example/
    original/001.obj
    subd0/001.obj
    subd1/001.obj
    subd2/001.obj
    subd3/001.obj
```

For custom data, simplify a reference mesh and generate supervision using successive self-parameterization (SSP). A C++ implementation is available under `08_subdiv_remesh` in the [Surface Multigrid repository](https://github.com/HTDerekLiu/surface_multigrid_code).

```bash
python preprocessdata.py
```

This reads `data_meshes/example/`, normalizes all levels using the reference bounding-box center and half its longest side, and writes `data_PKL/train.pkl`. Both training stages use this cache.

## Training
```bash
python writeHyperparam.py 
python train.py ./results
```
To run the stages separately, execute these commands in order.

```bash
python train.py jobs/example --stage1-only
python train.py jobs/example --stage2-only
```

## Testing and decoding

Decode the supplied pretrained example with an explicit stage selection.

```bash
python test.py results data_meshes/example/subd0/001.obj --stage 1
python test.py results data_meshes/example/subd0/001.obj --stage 2
python test.py results data_meshes/example/subd0/001.obj --stage all
## Acknowledgments
```
--stage 1` exports the spectral subdivision hierarchy and requires the Stage 1 parameters.` `--stage 2` and `--stage all` both load the two parameters and export the hierarchy with normal displacement applied at the terminal level.

## Acknowledgments

Parts of this implementation are adapted from [Neural Subdivision](https://github.com/HTDerekLiu/neuralSubdiv). The source is distributed under the [Mozilla Public License 2.0](LICENSE).
