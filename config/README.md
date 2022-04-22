## How to use config files

### Training config

We provide some examplar config files for SphereFace, SphereFace+ and SphereFace2 in the `config/train` directory:
- SphereFace on SFNet-20 (without BN): `vggface2_sfnet20_sphereface.yml`
- SphereFace+ on SFNet-20 (without BN): `vggface2_sfnet20_spherefaceplus.yml`
- SphereFace2 on SFNet-20 (without BN): `vggface2_sfnet20_sphereface2.yml`
- SphereFace2 one SFNet-64 (without BN): `vggface2_sfnet64_sphereface2.yml`

For example, to train a model with a specific config file `vggface2_sfnet20_sphereface.yml`, simply run

```console
CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/train/vggface2_sfnet20_sphereface.yml
```

To use four GPUs, simply modify `CUDA_VISIBLE_DEVICES=0,1` to `CUDA_VISIBLE_DEVICES=0,1,2,3` (with corresponding GPU id).

### Testing config

After finishing training a model, you will see a `project` folder under `$OPENSPHERE_ROOT`. The trained model is saved in the folder named by the job starting time, eg, `20220422_031705` for 03:17:05 on 2022-04-22.

To test a trained model, we provide testing config files for the combined validation set and IJB dataset.

For example, simply run the following commend to test a trained model (saved in `project/20220422_031705`) on IJB-B:

```console
CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/test/ijbb.yml --proj_dir project/20220422_031705
```
### Customize config files

Please take a look at the examples given in `config/train` and `config/test`.
