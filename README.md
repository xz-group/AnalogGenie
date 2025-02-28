# AnalogGenie

A generative engine for automatic discovery of analog circuit topologies by representing circuits as Eulerian circuits and using a decoder-only transformer to predict the next device pin. 

## About this work

For more details, please refer to our **ICLR'25** paper: [_AnalogGenie: A Generative Engine for Automatic Discovery of Analog Circuit Topologies_](https://openreview.net/forum?id=jCPak79Kev)

## How to Use

### Environment Setup

This setup requires Anaconda. Run the following command below:

```bash
conda env create -f environment.yml
```

To activate the environment:

```bash
conda activate AnalogGenie
```

### Dataset

Check [data_categorization.md](Dataset/data_categorization.md) file for dataset categorization

### Data Preprocessing

Convert SPICE netlist to adjacency matrix (SPICE2GRAPH_full.py will result in a more expressive but dense graph)

```
python SPICE2GRAPH_compress.py
```

Convert adjacency matrix to Eulerian circuit and perform augmentation

```
python Augmentation.py
```

Stack Eulerian circuits to NumPy array for training and validation

```
python Stack.py
```

### Pretraining and Inference

Perform pretraining

```
python Pretrain.py
```

Perform inference and generate circuits

```
python Inference.py
```

## Citation

If you use this framework for your research, please cite our [ICLR'25 paper](https://openreview.net/forum?id=jCPak79Kev):

```
@inproceedings{
gao2025analoggenie,
title={AnalogGenie: A Generative Engine for Automatic Discovery of Analog Circuit Topologies},
author={Jian Gao and Weidong Cao and Junyi Yang and Xuan Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=jCPak79Kev}
}
```

## Contact Information

If you have any questions regarding using this framework, please feel free to contact us at [gao.jian3@northeastern.edu](mailto:gao.jian3@northeastern.edu).

## Version History

* 0.1
  * Initial Release

## License

This framework is licensed under the `MIT` License - see the [LICENSE.md](LICENSE) file for details
