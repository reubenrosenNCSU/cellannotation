
# Automatic Cell Annotation Tool

The primary objective is to enable biologists with little to no programming experience to deploy and use ML models for object detection purposes. A cell detection tool (mainly SGN and MADM) which allows a user to use object detection models to detect cells in a microscope image. 




## Features

- Automatically detects SGN and MADM cells
- Real time detection of cell count
- You can add new annotations or remove the detected annotations after detection is completed
- You can fine tune the model, either by uploading your own images and  corresponding annotations, or by using the images you have previously used on this tool.
- Crop, zoom, brightness and contrast changes to adjust view of image.




## Deployment

To deploy this project run

```bash
  python app.py
```

and then open
```bash
127.0.0.1:5000/static/index.html
```

you can also use your host static IP so that multiple computers on the same network can access it. Make sure firewall permissions are granted. You  can install the dependencies using the following
```bash
pip install -r requirements.txt
```
alternatively you can use the requirements.yml file. it is a copy of the conda environment I had used in development.
```bash
conda env create -f requirements.yml
```
