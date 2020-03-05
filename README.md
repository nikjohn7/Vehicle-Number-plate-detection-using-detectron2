# Vehicle Number plate detection using detectron2

This project demonstrates the use of [Detectron2 by Facebook AI Research](https://github.com/facebookresearch/detectron2) to detect and isolate number plates from vehicles.

Dataset: [Custom dataset](https://drive.google.com/file/d/1u1VNPrDPP6AePoiYESldTBepFaamrMbY/view)


## Technology stacks and tools used
* [Detectron2](https://github.com/facebookresearch/detectron2) by Facebook AI Research
* [PyTorch](https://pytorch.org) deep learning framework.
* [Cython](https://github.com/cython/cython)
* [Google Colab](https://colab.research.google.com/)
* [LabelMe](https://github.com/wkentaro/labelme)
* [CocoAPI](https://github.com/cocodataset/cocoapi)


## Installation

For PyTorch, use the command

```bash
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

For Cython, use the command

```bash
pip3 install Cython
```

Then clone the detectron repo and cocoapi repo, using the commands

```bash
pip install -U 'git+https://github.com/facebookresearch/fvcore.git'
cd detectron2_repo
python setup.py build develop
git clone https://github.com/philferriere/cocoapi
cd .\cocoapi\PythonAPI\
```

Then unzip the ```plates_coc.zip``` in the main folder


## Usage
First train the model using
```python
python .\training.py
```
Then view the inferences by running
```python
python .\inference.py   
```
There might be some packages missing in case you're not running this on colab. Mostly pertaining to missing packages, which can easily be identified

You can access a pretrained model [here](https://drive.google.com/file/d/1hMCczENeCLd-jq7KGW9ak8B-QkWVUoXg/view), in case you want to avoid training 

## Results
Training.py
![Annotation 2020-03-05 201511](https://user-images.githubusercontent.com/29889429/75994820-88172880-5f21-11ea-9b20-6029af2e7927.png)

Outputs of some images and their extracted plates
![Annotation 2020-03-06 002910](https://user-images.githubusercontent.com/29889429/76015896-e94ef400-5f41-11ea-9c05-9222a117b70b.png)
![Annotation 3](https://user-images.githubusercontent.com/29889429/76015899-eb18b780-5f41-11ea-9962-f7296594e8bb.png)
![Annotation 4](https://user-images.githubusercontent.com/29889429/76015901-ec49e480-5f41-11ea-985c-f68e74c47c79.png)

Considering that the dataset had a meager 414 images with annotations, the results were decent

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update the tests as appropriate.

## References
* [https://medium.com/towards-artificial-intelligence/number-plate-detection-recognition-detectron-v2-5ddad2a532d0](https://medium.com/towards-artificial-intelligence/number-plate-detection-recognition-detectron-v2-5ddad2a532d0)

* [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
